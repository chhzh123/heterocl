/*!
 *  Copyright (c) 2020 by Contributors
 * \file stream_inference.cc
 * \brief mutate ir for scheduling streaming ops
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <unordered_map>
#include "./ir_util.h"

namespace TVM {
namespace ir {

using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;

struct IoInfo {
    DeviceType    dev_type;
    StorageType   storage_type; 
    int           mem_port{-1};
    StreamType    stream_type;
    int           channel_depth{-1}; 
};

// The stream information of a buffer
struct StreamInfo {
    vector<int>    index_array;
    vector<int>    depth_array;
    int            max_consumers{0}; // for producer buffer
};
    
inline Expr Type2Expr(const Type& t) {
  if (t.code()  == Type::Handle) 
    return StringImm::make("handle");
  std::ostringstream os;
  os << t;
  return StringImm::make(os.str());
}

// Substitute the target buffer consumers with channel buffers
class NewChannelGathers final : public IRMutator {
 public: 
  NewChannelGathers(vector<int> _index_array,
    string _target_buffer_name,
    StreamInfo _target_buffer_stream_info,
    unordered_map<int, VarExpr>&  _channel_index_to_new_buffers) :
    index_array(_index_array), 
    target_buffer_name(_target_buffer_name),
    target_buffer_stream_info(_target_buffer_stream_info),
    channel_index_to_new_buffers(_channel_index_to_new_buffers) {}

  Stmt Mutate(Stmt stmt) final {
    if (!hit_target_channel_load) {
        Stmt ret = IRMutator::Mutate(stmt);

        // Add temp to save value before the statement
        if (hit_target_channel_load) {
            if (search_first_stmt_with_target == 0) {
                HCL_DEBUG(2) << "Insert streaming channel reader of "
                    << target_buffer_name << " before "
                    << "the first Stmt consumer:"; 
                HCL_DEBUG(2) << "    " << ret;

                // Loading data from the channel 
                // TODO: support multiple index case
                auto index = index_array[0]; 
                auto target_load_op = target_load_expr.as<Load>();
                CHECK(target_load_op);
                CHECK(channel_index_to_new_buffers.count(index));
                VarExpr channel_buf(channel_index_to_new_buffers[index].node_); 
                Expr new_load = Load::make(target_load_op->type, 
                    channel_buf, target_load_op->index, target_load_op->predicate);

                Stmt s = Store::make(new_var, new_load, 0, 
                    UIntImm::make(UInt(1), 1));
                ret = Block::make(s, ret);
                ret = Allocate::make(new_var, target_load_op->type, {1}, 
                        make_const(Bool(target_load_op->type.lanes()), true), ret);
                ret = AttrStmt::make(new_var, attr::storage_scope, 
                        StringImm::make("global"), ret);
            }
            search_first_stmt_with_target++;
        }
        return ret;
    }
    return IRMutator::Mutate(stmt);
  }

  // Here we should check whether the accessing 
  // sequence is the same as the producer
  Expr Mutate_(const Load* op, const Expr& e) {
    auto name = op->buffer_var.get()->name_hint;
    if (name == target_buffer_name) {
        hit_target_channel_load = true;
        CHECK(new_var.defined());
        target_load_expr = e;
        return Load::make(op->type, new_var, 0, op->predicate);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt SubstituteBufferLoads(Stmt s) {
    new_var = VarExpr(target_buffer_name + ".temp");
    s = Mutate(s);
    return s;
  }

  vector<int> index_array;
  string target_buffer_name;
  StreamInfo  target_buffer_stream_info;
  unordered_map<int, VarExpr>& channel_index_to_new_buffers;

  VarExpr new_var;
  Expr target_load_expr;
  bool hit_target_channel_load{false};
  int search_first_stmt_with_target{0};
};

// Create new channels 
class NewChannelCreators final : public IRMutator {
 public: 
  NewChannelCreators(vector<int> _index_array,
    string _target_buffer_name,
    StreamInfo _target_buffer_stream_info,
    unordered_map<int, VarExpr>&  _channel_index_to_new_buffers,
    unordered_map<string, Type> _dtype) :
    index_array(_index_array), 
    target_buffer_name(_target_buffer_name),
    target_buffer_stream_info(_target_buffer_stream_info),
    channel_index_to_new_buffers(_channel_index_to_new_buffers),
    dtype(_dtype) {}

  Stmt Mutate_(const Store* op, const Stmt& s) {
    auto name = op->buffer_var.get()->name_hint;

    // Use a temp value to store the value into a temp
    // There should only be a signle store for the target buffer
    if (name == target_buffer_name) {
        CHECK(!buffer_created) << "Failure: trying to stream a tensor that "
            << "has been written for multiple times...";
        HCL_DEBUG(2) << "Found target buffer store of " << name;

        buffer_created = true;
        VarExpr temp(name + ".temp");
        CHECK(dtype.count(target_buffer_name));
        auto type = dtype[target_buffer_name];
        Stmt stmt = Store::make(temp, op->value, 0, op->predicate);
        
        // Create buffers for vars in index array
        for (size_t k = 0; k < index_array.size(); k++) {
            auto index = -1 * index_array[k];
            auto new_name = name + ".pipe." + std::to_string(index);
            VarExpr new_channel_buffer(new_name);
            channel_index_to_new_buffers[index] = new_channel_buffer;
            HCL_DEBUG(2) << "Adding new buffer " << new_name
                << " for channel #" << index << "...";

            // Create store nodes to save the temp var
            Expr e = Load::make(type, temp, 0, op->predicate);
            Stmt s = Store::make(new_channel_buffer, e, op->index, op->predicate);
            stmt = Block::make(stmt, s);
        }

        // Write back to origina buffer if some
        // consumers still reads from it
        if (write_back) {
          Expr e = Load::make(type, temp, 0, op->predicate);
          Stmt s = Store::make(op->buffer_var, e, op->index, op->predicate);
          stmt = Block::make(stmt, s);
        }

        stmt = Allocate::make(temp, type, {1}, 
                make_const(Bool(type.lanes()), true), stmt);
        stmt = AttrStmt::make(temp, attr::storage_scope, 
                StringImm::make("global"), stmt);
        return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt CreateBuffers(Stmt stmt, Array<Expr> shape) {
    write_back = ((int)index_array.size() 
        == target_buffer_stream_info.max_consumers) ? false : true;
    Stmt s = Mutate(stmt);

    // Add buffer allocation nodes
    // at the beginning of the producer stage (stream_scope attr)
    for (auto index : index_array) {
        
        index *= -1;
        CHECK(channel_index_to_new_buffers.count(index)) << index;
        VarExpr buf(channel_index_to_new_buffers.at(index).node_); 
        auto channel_index = index; 

        int channel_depth = -1;
        auto index_array = target_buffer_stream_info.index_array;
        for (size_t k = 0; k < index_array.size(); k++) {
            if (index_array[k] == channel_index) {
              channel_depth = target_buffer_stream_info.depth_array[k];
            }
        }
        CHECK(channel_depth != -1);
        CHECK(dtype.count(target_buffer_name));
        Type type = dtype[target_buffer_name];

        Stmt attr = StreamStmt::make(
              buf, IntImm::make(Int(32), channel_index), 
              StreamType::FIFO, channel_depth, Array<Expr>(), Array<Expr>()); 
        Array<Stmt> attrs = { attr };
        s = Allocate::make(buf, type, shape, 
                   make_const(Bool(type.lanes()), true), s, attrs, Expr(), string());
        s = AttrStmt::make(buf, attr::storage_scope, 
                StringImm::make("global"), s);
    }

    return RemoveNoOp(s);
  }

  vector<int> index_array;
  string target_buffer_name;
  StreamInfo  target_buffer_stream_info;
  unordered_map<int, VarExpr>& channel_index_to_new_buffers;
  unordered_map<string, Type> dtype;

  bool buffer_created{false};
  bool write_back;
  
};

// Mutate the Allocate Stmt and add StreamStmt into the attr 
class AllocateAttrDecorator final : public IRMutator {
 public: 
  AllocateAttrDecorator(
    unordered_map<string, vector<int> > _global_channel_trace,
    unordered_map<string, StreamInfo>  _inter_stage_channels,
    unordered_map<string, Type> _dtype,
    unordered_map<string, Array<Expr> > _shape)
    : global_channel_trace(_global_channel_trace),
      inter_stage_channels(_inter_stage_channels), dtype(_dtype), shape(_shape) {}

  // Add StreamStmt as attributes to stream_scoped Allocate
  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();

    string name = op->buffer_var.get()->name_hint;
    if (global_channel_trace.count(name)) {
      HCL_DEBUG(2) << "Found Streaming Channel " << name;
      auto params = global_channel_trace[name];
      int channel_index = params[0];
      int channel_depth = params[1];
      Stmt attr = StreamStmt::make(
            op->buffer_var, IntImm::make(Int(32), channel_index), 
            StreamType::FIFO, channel_depth, Array<Expr>(), Array<Expr>()); 
      Array<Stmt> attrs = op->attrs;
      attrs.push_back(attr);
      return Allocate::make(op->buffer_var, op->type, op->extents,
                            op->condition, op->body, attrs,
                            op->new_expr, op->free_function);
    }
    return stmt;
  }

  // Mutate the stage body (with in the producer)
  // 1. Add new buffers (decorated with StreamStmt) as channels
  // 2. Add temp value to store the read value from prodoucer buffer
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    if (op->attr_key == attr::stream_attrs) {
        VarExpr var(op->node.node_);
        string buffer_name = var.get()->name_hint;
        Stmt body = op->body;

        CHECK(op->value.as<IntImm>());
        int index =  op->value.as<IntImm>()->value;
        HCL_DEBUG(2) << "Pushing channel index " << index 
            << " into array...";
        vector<int> index_array = {index};
        while (auto attr = body.as<AttrStmt>()) {
            if (attr->attr_key != attr::stream_attrs) {
                body = attr->body;
                continue;
            }
            CHECK(attr->value.as<IntImm>());
            int attr_index =  attr->value.as<IntImm>()->value;
            CHECK(attr_index * index_array.back() > 0) 
                << "Tensor " << buffer_name << " cannot be read and written "
                << "at the same time";
            HCL_DEBUG(2) << "Pushing channel index " << attr_index 
                << " into array...";
            index_array.push_back(attr_index);
            body = attr->body;
        }

        CHECK(inter_stage_channels.count(buffer_name));
        auto info = inter_stage_channels[buffer_name];
        // Producers nested attrs
        // 1. Create new buffers (attributed with StreamStmt)
        if (index_array.back() < 0) {
            HCL_DEBUG(2) << "Creating channel buffers on the producer side...";
            NewChannelCreators ncc(index_array, buffer_name, info, 
                channel_index_to_new_buffers, dtype);
            CHECK(shape.count(buffer_name));
            auto buf_shape = shape[buffer_name];
            return ncc.CreateBuffers(body, buf_shape);

        // Consumers nested attrs
        // 1. Used the buffers created by producers
        } else {
            NewChannelGathers ncg(index_array, buffer_name, info,
                channel_index_to_new_buffers);
            return ncg.SubstituteBufferLoads(body);
        }
    }
    return IRMutator::Mutate_(op, s);
  }

  unordered_map<string, vector<int> > global_channel_trace;
  unordered_map<string, StreamInfo>  inter_stage_channels;
  unordered_map<int, VarExpr>        channel_index_to_new_buffers;
  unordered_map<string, Type> dtype;
  unordered_map<string, Array<Expr> > shape;
};


// 1. Substitute old buffer with new buffers (i.e. the buffers
//    defined as kernel function arguments)
// 2. If the tensor is moved to host from device, remove the on-chip
//    buffer allocation and use a function arg buffer to replace it
class SubstituteBuffers final : public IRMutator {
 public: 
  SubstituteBuffers(unordered_map<const Variable*, VarExpr>& _vmap,
    unordered_map<string, VarExpr>& _remove)
  : vmap(_vmap), remove(_remove) {}

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();

    string name = op->buffer_var.get()->name_hint;
    if (remove.count(name)) {
      HCL_DEBUG(2) << "Lifting buffer (alloc) " << name;
      lifted_buffers.push_back(op->buffer_var);
      return op->body;
    }
    return stmt;
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    if (vmap.count(op->buffer_var.get())) {
        HCL_DEBUG(2) << "Substituting buffer (load) " << op->buffer_var;
        VarExpr new_var(vmap[op->buffer_var.get()].node_);
        return Load::make(op->type, new_var, op->index, op->predicate);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    Expr value = this->Mutate(op->value);
    string name = op->buffer_var.get()->name_hint;
    if (remove.count(name))  {
      HCL_DEBUG(2) << "Substituting buffer (store) " << name;
      VarExpr new_var(remove[name].node_);
      return Store::make(new_var, value, op->index, op->predicate);
    }
    if (vmap.count(op->buffer_var.get())) {
        HCL_DEBUG(2) << "Substituting buffer (store) " << op->buffer_var;
        VarExpr new_var(vmap[op->buffer_var.get()].node_);
        return Store::make(new_var, value, op->index, op->predicate);
    }
    return Store::make(op->buffer_var, value, op->index, op->predicate);
  }

  
  Stmt Mutate_(const Partition* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Partition>();
    if (vmap.count(op->buffer_var.get())) {
      HCL_DEBUG(2) << "Substituting buffer (partition) " << op->buffer_var;
      VarExpr new_var(vmap[op->buffer_var.get()].node_);
      return Partition::make(new_var, op->dim, op->factor, op->partition_type);
    } else {
      return stmt;
    }
  }

  unordered_map<const Variable*, VarExpr>& vmap;
  unordered_map<string, VarExpr>& remove;
  vector<VarExpr> lifted_buffers;

};

// 1. Create the KernelDef Stmt for device function by 
//    allocating the arg for IO args. For those ExternOpNode
//    output moved to host, we need to deallocate the buffers 
// 2. Create KernelStmt (i.e. dev function call)
class KernelDefCreator final : public IRMutator {
 public: 
  KernelDefCreator(unordered_map<string, IoInfo>& _dev_io_info,
      unordered_map<string, Array<Expr> >& _shape,
      unordered_map<string, Type>& _dtype)
  : dev_io_info(_dev_io_info), shape(_shape), dtype(_dtype) {}

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    string target_name = op->buffer_var.get()->name_hint;
    if (target_name == "test") {
      HCL_DEBUG(2) << "Removed unused var " << target_name;
      return this->Mutate(op->body);
    }
    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::device_scope) {

      // Reconstruct the dev scope function
      if (!op->node.defined()) { 
        Array<Var> undefs = UndefinedVars(op->body, Array<Var>());
        unordered_map<string, IoInfo> dev_io_copy = dev_io_info;

        // Buffers to substitute
        unordered_map<const Variable*, VarExpr> vmap;
        Array<VarExpr>      kernel_def_new_vars;
        Array<Expr>         kernel_stmt_vars;
        vector<Expr>        kernel_stmt_annotate_values;
        Array<Array<Expr> > shapes, attributes; 
        Array<Expr> types;
        Array<FunctionRef> placeholders;

        for (auto& v : undefs) {
          string name = v.get()->name_hint;
          CHECK(dev_io_copy.count(name)) << "Cannot find data placement information "
            << "of tensor " << name << ". Make sure it is moved by .to()...";

          CHECK(dtype.count(name) && shape.count(name));
          Type type = dtype[name];
          Array<Expr> arg_shape = shape[name];
          shapes.push_back(arg_shape);
          types.push_back(Type2Expr(type));

          // Prepare function IO attributes
          // Attributes to KernelDef Nodes
          Array<Expr> attr;
          auto io_attr = dev_io_copy.at(name);
          attr.push_back(StringImm::make(name));
          attr.push_back(IntImm::make(Int(32), static_cast<int>(io_attr.storage_type)));
          attr.push_back(IntImm::make(Int(32), io_attr.mem_port));
          attr.push_back(IntImm::make(Int(32), static_cast<int>(io_attr.stream_type)));
          attr.push_back(IntImm::make(Int(32), io_attr.channel_depth));
          attributes.push_back(attr);

          // Create new buffers to replace old buffers
          Var old_var(v.node_);
          VarExpr new_var(name);
          Operation op = PlaceholderOpNode::make(name, arg_shape, type);
          placeholders.push_back(op);
    
          vmap[old_var.get()] = VarExpr(new_var.node_);
          kernel_def_new_vars.push_back(new_var);
          kernel_stmt_vars.push_back(old_var);

          string value = std::to_string(static_cast<int>(io_attr.dev_type)) + ":" +
                std::to_string(static_cast<int>(io_attr.storage_type)) + ":" + 
                std::to_string(io_attr.mem_port) + ":" + std::to_string(static_cast<int>(io_attr.stream_type)) + ":" + 
                std::to_string(io_attr.channel_depth);
          kernel_stmt_annotate_values.push_back(StringImm::make(value));

          dev_io_copy.erase(name);
        }

        // Buffers to be lift atop kernel function call
        unordered_map<string, VarExpr> remove;
        for (auto& kv : dev_io_copy) {
          string name = kv.first;

          CHECK(dtype.count(name) && shape.count(name));
          Type type = dtype[name];
          Array<Expr> arg_shape = shape[name];
          shapes.push_back(arg_shape);
          types.push_back(Type2Expr(type));

          // Prepare function IO attributes
          Array<Expr> attr;
          auto io_attr = kv.second;
          attr.push_back(StringImm::make(name));
          attr.push_back(IntImm::make(Int(32), static_cast<int>(io_attr.storage_type)));
          attr.push_back(IntImm::make(Int(32), io_attr.mem_port));
          attr.push_back(IntImm::make(Int(32), static_cast<int>(io_attr.stream_type)));
          attr.push_back(IntImm::make(Int(32), io_attr.channel_depth));
          attributes.push_back(attr);

          VarExpr new_var(name);
          remove[name] = new_var;
          kernel_def_new_vars.push_back(new_var);

          Operation op = PlaceholderOpNode::make(name, arg_shape, type);
          placeholders.push_back(op);
        };
        
        // Replace buffers
        SubstituteBuffers sb(vmap, remove);
        Stmt body = sb.Mutate(op->body);

        // Create KernelDef Stmt based on body
        Stmt kernel = KernelDef::make(kernel_def_new_vars, shapes, types, 
                          placeholders, body, UIntImm::make(UInt(1), 1),
                          UInt(32), "test", attributes); 
        kernel_defs_.push_back(kernel);

        // Buffer lifting and return KernelStmt
        CHECK(dev_io_copy.size() == sb.lifted_buffers.size());
        for (auto& var : sb.lifted_buffers) {
            Expr new_arg(var.node_);
            kernel_stmt_vars.push_back(new_arg);

            CHECK(dev_io_copy.count(var.get()->name_hint));
            auto io_attr = dev_io_copy.at(var.get()->name_hint);
            string value = std::to_string(static_cast<int>(io_attr.dev_type)) + ":" +
                  std::to_string(static_cast<int>(io_attr.storage_type)) + ":" + 
                  std::to_string(io_attr.mem_port) + ":" + std::to_string(static_cast<int>(io_attr.stream_type)) + ":" + 
                  std::to_string(io_attr.channel_depth);
            kernel_stmt_annotate_values.push_back(StringImm::make(value));
        }

        // Prepare the annotate keys and values 
        Array<Expr> keys, values;
        for (size_t k = 0; k < kernel_stmt_vars.size(); k++) {
            keys.push_back(IntImm::make(Int(32), k));
            values.push_back(kernel_stmt_annotate_values[k]);
        }
        Stmt stmt = KernelStmt::make(kernel_stmt_vars, "test", keys, values);

        for (auto& var : sb.lifted_buffers) {
          string name = var.get()->name_hint;
          CHECK(dtype.count(name) && shape.count(name));
          Type type = dtype[name];
          Array<Expr> arg_shape = shape[name];
          stmt = Allocate::make(var, type, arg_shape, make_const(Bool(type.lanes()), true), stmt);
          stmt = AttrStmt::make(var, attr::storage_scope, StringImm::make("global"), stmt);
        }
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  // Replace device scope with KernelStmt
  // and block it with the newly generated KernelDef
  Stmt SplitScope(Stmt stmt) {
    Stmt s = Mutate(stmt);
    for (auto& k : kernel_defs_) s = Block::make(k, s);
    return RemoveNoOp(s);
  }

  unordered_map<string, IoInfo> dev_io_info;
  unordered_map<string, Array<Expr>> shape;
  unordered_map<string, Type> dtype;
  vector<Stmt> kernel_defs_;
};


// Collect the host-device information 
// 1. The IO information that is used to create the KernelDef function 
//    and the KernelStmt for calling the device function  
// 2. Buffer type and shape information 
// 3. Whether a buffer is passed from top level (storage_scope)
class StreamInfoCollector final : public IRMutator {
  public: 
    StreamInfoCollector(Array<NodeRef>& api_args) {
    for (size_t i = 0; i < api_args.size(); i++) { 
      if (const Variable* v = api_args[i].as<Variable>()) {
        top_arg_names.insert(v->name_hint);

      } else if (auto buf = api_args[i].as<BufferNode>()) {
        CHECK(buf->data.as<Variable>());
        top_arg_names.insert(buf->name); 

        shape_[buf->data.get()->name_hint] = buf->shape;
        dtype_[buf->data.get()->name_hint] = buf->dtype;
      }
    }
  };

  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    auto v = op->buffer_var.get();
    auto name = v->name_hint; 
    // Save shape and dtype information
    shape_[name] = op->extents;
    dtype_[name] = op->type;
    return IRMutator::Mutate_(op, s);
  }

  // Record the IO interface information
  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::io_interface) {
      CHECK(op->value.as<StringImm>());
      string s = op->value.as<StringImm>()->value;

      size_t pos = 0;
      string delimiter = ":";
      string token;
      vector<int> numbers;
      while ((pos = s.find(delimiter)) != string::npos) {
          token = s.substr(0, pos);
          numbers.push_back(std::stoi(token));
          s.erase(0, pos + delimiter.length());
      }

      // Memory type, MemPort, StreamType, ChannelDepth
      numbers.push_back(std::stoi(s));
      CHECK(numbers.size() == 5);
      IoInfo io_info;
      io_info.dev_type      = static_cast<DeviceType>(numbers[0]);
      io_info.storage_type  = static_cast<StorageType>(numbers[1]); 
      io_info.mem_port      = numbers[2];
      io_info.stream_type   = static_cast<StreamType>(numbers[3]); 
      io_info.channel_depth = numbers[4];

      VarExpr var(op->node.node_);
      string name = var.get()->name_hint; 
      dev_io_info[name] = io_info;

      return this->Mutate(op->body);

    // The global channel (e.g. Intel channels)
    } else if (op->attr_key == attr::stream_scope) {
      CHECK(op->value.as<StringImm>());
      string s = op->value.as<StringImm>()->value;

      size_t pos = 0;
      string delimiter = ":";
      string token;
      vector<int> numbers;
      while ((pos = s.find(delimiter)) != string::npos) {
          token = s.substr(0, pos);
          numbers.push_back(std::stoi(token));
          s.erase(0, pos + delimiter.length());
      }

      // Channel index, channel depth  
      numbers.push_back(std::stoi(s));
      CHECK(numbers.size() == 2);
      VarExpr var(op->node.node_);
      string name = var.get()->name_hint; 
      global_channel_trace[name] = numbers;
      return this->Mutate(op->body);

    // The tensor to be streamed (inter-stage)
    // Need to create channels explictly  
    } else if (op->attr_key == attr::stream_attrs) {
      CHECK(op->value.as<StringImm>());
      string s = op->value.as<StringImm>()->value;

      size_t pos = 0;
      string delimiter = ":";
      string token;
      vector<int> numbers;
      while ((pos = s.find(delimiter)) != string::npos) {
          token = s.substr(0, pos);
          numbers.push_back(std::stoi(token));
          s.erase(0, pos + delimiter.length());
      }

      // Channel index, channel depth, is_producer 
      numbers.push_back(std::stoi(s));
      CHECK(numbers.size() == 4);
      VarExpr var(op->node.node_);
      string name = var.get()->name_hint; 

      int  channel_index    = numbers[0];
      int  channel_depth    = numbers[1];
      bool is_producer      = (numbers[2] == 1) ? true : false;
      int  max_consumers    = numbers[3];

      // Information processing 
      // 1. If # stream channel <  # consumers. Then 
      //    we need to allocate channels for each streaming pair
      //    and finally written the value back to original buffer
      // 2. If # stream channel == # consumers. Same
      //    case. we do not need to write data back to original buffers
      StreamInfo info;
      if (inter_stage_channels.count(name)) {
        info = inter_stage_channels.at(name);
      }

      if (is_producer) {
        info.index_array.push_back(channel_index);
        info.depth_array.push_back(channel_depth);
        info.max_consumers = max_consumers;
        channel_index *= -1;
      }

      inter_stage_channels[name] = info;
      return AttrStmt::make(op->node, attr::stream_attrs,
            IntImm::make(Int(32), channel_index), this->Mutate(op->body));
    }

    return IRMutator::Mutate_(op, s);
  }

  // Mark the global scoped buffer in KernelStmt 
  Stmt Mutate_(const KernelStmt* op, const Stmt& s) final {
    int pos = 0;
    for (auto arg : op->args) {
      auto name = arg.as<Variable>()->name_hint;
      if (top_arg_names.find(name) != top_arg_names.end())
        global_buffer_trace[op->name].insert(pos);
      pos += 1;
    }
    return IRMutator::Mutate_(op, s);
  }

  unordered_set<string> top_arg_names;
  unordered_map<string, Array<Expr> > shape_;
  unordered_map<string, Type> dtype_;
  unordered_map<string, IoInfo> dev_io_info;

  unordered_map<string, unordered_set<int> > global_buffer_trace;
  unordered_map<string, vector<int> > global_channel_trace;
  unordered_map<string, StreamInfo>  inter_stage_channels;
};


class StoreToStreamStmtConverter final : public IRMutator {
  public: 
    StoreToStreamStmtConverter(
        const string& target,
        const ir::StreamType& type,
        const VarExpr& channel_buf,
        const int channel_depth,
        int channel_index,
        const Array<Expr> shape,
        unordered_map<const Variable*, Expr>& range) 
      : target_(target), type_(type), channel_buf_(channel_buf),
        channel_depth_(channel_depth), channel_index_(channel_index), 
        shape_(shape), range_(range) {} 

    Stmt Mutate_(const Store* op, const Stmt& s) {
      Expr index = op->index;
      Expr value = this->Mutate(op->value);
      string target_name = op->buffer_var.get()->name_hint;
      if (target_name == target_) {
        Array<Expr> keys, values;
        // push channel and access information 
        keys.push_back(StringImm::make("index"));
        values.push_back(index);
        keys.push_back(StringImm::make("channel"));
        values.push_back(IntImm::make(Int(32), channel_index_));
        return StreamStmt::make(VarExpr(channel_buf_.node_), value, 
                                type_, channel_depth_, keys, values); 
      } else {
        return Store::make(op->buffer_var, value, 
                           index, op->predicate);
      }
    }

  private:
    const string target_;
    const ir::StreamType type_;
    const VarExpr& channel_buf_;
    const int channel_depth_;
    const int channel_index_;
    const Array<Expr> shape_;
    unordered_map<const Variable*, Expr>& range_;
};

class LoadToStreamExprConverter final : public IRMutator {
  public: 
    LoadToStreamExprConverter(
        const string& target,
        const ir::StreamType& type,
        const VarExpr& channel_buf,
        const int channel_depth,
        int channel_index, 
        const Array<Expr> shape,
        unordered_map<const Variable*, Expr>& range) 
      : target_(target), type_(type), channel_buf_(channel_buf),
        channel_depth_(channel_depth), channel_index_(channel_index), 
        shape_(shape), range_(range) {} 

    // record axis to mutate streaming sender 
    Stmt Mutate_(const For* op, const Stmt& s) {
      Stmt stmt = IRMutator::Mutate_(op, s);
      if (found) // in the right track
        loop_vars.push_back(op->loop_var.get());
      return stmt;
    }

    // single load repalcement 
    Expr Mutate_(const Load* op, const Expr& e) {
      Expr index = op->index;
      string target_name = op->buffer_var.get()->name_hint;
      if (target_ == target_name) {
        Array<Expr> keys, values;
        // push channel and access information 
        keys.push_back(StringImm::make("index"));
        values.push_back(std::move(op->index));

        keys.push_back(StringImm::make("channel"));
        values.push_back(IntImm::make(Int(32), channel_index_));
        return StreamExpr::make(op->type, VarExpr(channel_buf_.node_), 
                                type_, channel_depth_, keys, values);
      } else {
        return Load::make(op->type, op->buffer_var, 
                          index, op->predicate);
      }
   }
    std::vector<const Variable*> loop_vars;

  private:
    bool found{false};           // found tagret load op 
    const string target_;   // stream variable name 
    const ir::StreamType type_;  // stream types (fifo, channel, pipe)
    const VarExpr& channel_buf_; // streaming channel buffer
    const int channel_depth_;    // stream channel depth (no less than 0)
    const int channel_index_;    // stream channel index (share no more than 2 agents)
    const Array<Expr> shape_;    // shape array of target load op
    unordered_map<const Variable*, Expr>& range_; // range map of IterVar
};

// block kernel body with reuse buffer
Stmt BufferInserter(Stmt stmt, /*original extern op body*/
                    Array<Expr> shape, /*target buffer shape*/ 
                    const VarExpr& target, /*target load & store buf*/
                    const VarExpr& c_buf, /*channel buffer*/
                    bool load_mode, /*load or store mode*/
                    StreamType type, int channel_depth) {
  // compute indices for load / store
  std::vector<Expr> indices;
  std::vector<VarExpr> loop_vars;
  for (size_t i = 0; i < shape.size(); i++) {
    VarExpr iter("buf_" + std::to_string(i));
    indices.push_back(iter);
    loop_vars.push_back(iter);
  }
  Expr index = FlattenIndices(indices, shape); 
  
  if (load_mode) { // local buffer reading from stream channel  
    Expr stream = StreamExpr::make(target->type,
                                   VarExpr(c_buf.node_),
                                   type, channel_depth);
    // store op initialized with variable node
    Stmt for_stmt = Store::make(VarExpr(target.node_),
                                stream, index,
                                UIntImm::make(UInt(1), 1));
    
    auto type = ForType::Serial;
    for (size_t j = 0; j < shape.size(); j++) {
      auto iter = loop_vars[j];
      // DMA burst loading from sys memory  
      if (j == shape.size() - 1) type = ForType::Pipelined; 
      for_stmt = For::make(VarExpr(iter.node_), 0, shape[j],
                           type, DeviceAPI::None, for_stmt);
    }
    stmt = Block::make(for_stmt, stmt); 

  } else { // multiple stores : sending at end 
    Expr load = Load::make(target->type,
                           VarExpr(target.node_), index, 
                           UIntImm::make(UInt(1), 1));
    Stmt for_stmt = StreamStmt::make(VarExpr(c_buf.node_),
                                     load, type, channel_depth);

    auto type = ForType::Serial;
    for (size_t j = 0; j < shape.size(); j++) {
      auto iter = loop_vars[j];
      // DMA burst store to sys memory  
      if (j == shape.size() - 1) type = ForType::Pipelined; 
      for_stmt = For::make(VarExpr(iter.node_), 0, shape[j],
                           type, DeviceAPI::None, for_stmt);
    }
    stmt = Block::make(stmt, for_stmt); 
  }

  return stmt;
};

// collect access pattern for target vars
class AccessCollector : public ir::IRMutator {
 public:
  explicit AccessCollector(
      const VarExpr& target_buf, const Array<Expr>& shape,
      const unordered_map<const Variable*, Expr>& range,
      const string channel_name)
      : target_buf_(target_buf), shape_(shape), range_(range), 
        channel_name_(channel_name) {}

  // trace buffer allocation 
  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    string target_name = op->buffer_var.get()->name_hint;
    // whether the target buffer has been allocated 
    if (target_name == channel_name_) buf_alloc = true;
    return s;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    Expr value = this->Mutate(op->value);
    string target_name = op->buffer_var.get()->name_hint;
    // check if target buffer matches
    if (op->buffer_var.get() == target_buf_.get()) {
      store_num += 1; 
      store_var = VarExpr(op->buffer_var.node_);
      // check index access regularity 
      auto max_bound = Substitute(op->index, range_); 
      reg_store = is_zero(Simplify(max_bound - get_max(shape_)));
    }
    return s;
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    string target_name = op->buffer_var.get()->name_hint;
    // check if buffer matches
    if (op->buffer_var.get() == target_buf_.get()) {
      load_num += 1; 
      load_var = VarExpr(op->buffer_var.node_);
      // check index access regularity 
      auto max_bound = Substitute(op->index, range_); 
      reg_load = is_zero(Simplify(max_bound - get_max(shape_)));
    }
    return e;
  }

  int load_num{0};
  int store_num{0};
  VarExpr load_var;
  VarExpr store_var;
  bool reg_store{true};
  bool reg_load{true};
  bool buf_alloc{false};

 private:
  const VarExpr& target_buf_; /*stream variable buffer*/ 
  const Array<Expr>& shape_;  /*stream variable shape*/ 
  const unordered_map<const Variable*, Expr>& range_;
  const string channel_name_;

  Expr get_max(Array<Expr> shape) {
    Expr ret(shape[0]); 
    for (size_t i = 1; i < shape.size(); i++) ret *= shape[i]; 
    return Simplify(ret - 1); 
  }
};

// create streaming channels across loop iterations
class LoopbackMutator : public ir::IRMutator {
 public:
  explicit LoopbackMutator(
    const VarExpr& target_buf, const Array<Expr>& shape,
    const unordered_map<const Variable*, Expr>& range, 
    Type type)
  : target_buf_(target_buf), shape_(shape), 
    range_(range), type_(type) {} 

  // FIXME: buffer mismatch 
  Stmt Mutate_(const Store* op, const Stmt& s) {
    if (op->buffer_var->name_hint == target_buf_->name_hint) {
      if (store_count == 0) { 
        store_count += 1;
        CHECK(!temp_.defined());
        temp_ = VarExpr("temp_" + target_buf_->name_hint); 
        auto index = IntImm::make(Int(32), 0);
        Expr load_expr = Load::make(type_, 
                             temp_, index, op->predicate);
        save_stmt = Store::make(op->buffer_var, 
                        load_expr, op->index, op->predicate);

        Stmt stmt = Store::make(temp_, op->value, index, op->predicate);
        stmt = Allocate::make(temp_, type_, Array<Expr>(),
            make_const(Bool(type_.lanes()), true), stmt);
        stmt = AttrStmt::make(temp_, attr::storage_scope,
            StringImm::make("local"), stmt);
        return stmt;

      } else {
        store_count += 1;
        auto index = IntImm::make(Int(32), 0);
        return Store::make(temp_, op->value, index, op->predicate);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    if (op->buffer_var->name_hint == target_buf_->name_hint) {
      if (store_count > 0) { 
        auto index = IntImm::make(Int(32), 0);
        return Load::make(op->type, temp_, index, op->predicate);
      }
    }
    return e;
  }

  // create stream array
  Stmt Mutate_(const For* op, const Stmt& s) {

    if (op->body.as<For>() == nullptr) {
      Stmt stmt = this->Mutate(op->body);
      stmt = Block::make(stmt, save_stmt);
      return For::make(
          op->loop_var, op->min, op->extent, op->for_type,
          op->device_api, stmt, op->annotate_keys,
          op->annotate_values);

    } else {
      Stmt stmt = this->Mutate(op->body);
      return For::make(
          op->loop_var, op->min, op->extent, op->for_type,
          op->device_api, stmt, op->annotate_keys,
          op->annotate_values);
    }
  }

  private:
   const VarExpr& target_buf_;
   const Array<Expr>& shape_;
   const unordered_map<const Variable*, Expr>& range_;
   Type type_; 
   VarExpr temp_;
   int store_count{0};
   Stmt save_stmt;
};


// create local copy and sync with data copy 
class MultiLoadMutator : public IRMutator {
 public:
  explicit MultiLoadMutator(
    string& target,
    std::vector<VarExpr>& channels, Type type)
    : target_(target), channels_(channels), type_(type) {}

  Stmt Mutate(Stmt stmt) final {
    Stmt ret = IRMutator::Mutate(stmt);
    if (found && !alloc) { 
      for (auto& channel : channels_) {
        auto stream_expr = StreamExpr::make(type_, 
            VarExpr(channel.node_), StreamType::FIFO, 
            1, Array<Expr>(), Array<Expr>()); 

        auto store = Store::make(temp_, 
                stream_expr, Expr(0), const_true());
        ret = Block::make(store, ret);
      }
      ret = Allocate::make(temp_, type_, Array<Expr>(),
          make_const(Bool(type_.lanes()), true), ret);
      ret = AttrStmt::make(temp_, attr::storage_scope,
          StringImm::make("local"), ret);
      alloc = true;
    }
    return ret;
  }

  Expr Mutate_(const Load *op, const Expr& e) final {
    Expr index = op->index;
    string target_name = op->buffer_var.get()->name_hint;

    Stmt stmt;
    if (target_name == target_) {
      found = true;
      temp_ = VarExpr("temp_" + target_);
      return Load::make(op->type, temp_, index, op->predicate);
    } else {
      return Load::make(op->type, op->buffer_var, index, op->predicate);
    }
  }

 private:
  string& target_;
  std::vector<VarExpr>& channels_;
  Type type_;
  VarExpr temp_;
  bool found{false};
  bool alloc{false};
};

// create local copy and multiple streaming channels
class MultiCastMutator : public IRMutator {
 public:
  explicit MultiCastMutator(
    string& target,
    std::vector<VarExpr>& channels, Type type)
    : target_(target), channels_(channels), type_(type) {}

  Stmt Mutate_(const Store *op, const Stmt& s) final {
    Expr index = op->index;
    Expr value = this->Mutate(op->value);
    string target_name = op->buffer_var.get()->name_hint;
    if (target_name == target_) {
      VarExpr temp("temp");
      Stmt stmt = Store::make(temp, value, Expr(0), op->predicate);
      for (auto& channel : channels_) {
        auto stream_stmt = StreamStmt::make(
            VarExpr(channel.node_), temp, 
            StreamType::FIFO, 1, Array<Expr>(), Array<Expr>()); 
        stmt = Block::make(stmt, stream_stmt);
      }
      stmt = Allocate::make(temp, type_, Array<Expr>(),
          make_const(Bool(type_.lanes()), true), stmt);
      stmt = AttrStmt::make(temp, attr::storage_scope,
          StringImm::make("local"), stmt);
      return stmt;
    } else {
      return Store::make(op->buffer_var, value, 
                         index, op->predicate);
    }
  }

 private:
  string& target_;
  std::vector<VarExpr>& channels_;
  Type type_;

};


// 1. add annotation to kernel def node 
// 2. mutate the producer marked with .new 
// 3. remove defined but unused vars
class KernelAnnotator final : public IRMutator {
 public:
  KernelAnnotator(
    unordered_map<string, unordered_set<int>> map,
    unordered_map<string, Array<Expr>> mem_ports, 
    unordered_set<const Variable*>& unused_vars) :
    arg_scope_map_(map), mem_ports_(mem_ports), unused_vars_(unused_vars) {} 

  Stmt Mutate_(const KernelDef *op, const Stmt& s) final {
    Stmt body = this->Mutate(op->body);
    Array<Array<Expr>> channels = op->attributes;

    // insert annotation for top function 
    if (op->name == "test") {
      int count = 0;
      for (auto& arg : op->args) {
        auto name = arg->name_hint;
        // skip inner loop movement case 
        if (!mem_ports_.count(name)) {
          HCL_DEBUG(2) << "device function within loop or zerocopy mode";
          break;
        }
        auto dev_port = mem_ports_[name];
        CHECK(dev_port.size() == 4);
        auto direction = dev_port[3];
        // pos, channel index, depth, is_sedner, dev_type, mem_port
        Array<Expr> info = {
            count, /*arg position index*/ 
            -1,    /*arg streaming channel index*/ 
            -1,    /*streaming channel depth*/ 
            dev_port[3], /*if it is the producer*/ 
            dev_port[0], /*memory type*/ 
            dev_port[1], /*memory channel port*/
            dev_port[2], /*stream type*/
        };
        count = count + 1;
        channels.push_back(info);
      }
      return KernelDef::make(
                 op->args, op->arg_shapes, op->arg_types, 
                 op->arg_tensors, body, op->ret_void, 
                 op->ret_type, op->name, channels);
    }

    // mutate kernel def body 
    if (channels.size() > 0) {
      for (size_t i = 0; i < channels.size(); i++) {
        auto info = channels[i];
        CHECK(info.size() == 6);
        auto pos = info[0].as<IntImm>()->value;
        auto channel = info[1].as<IntImm>()->value;
        auto depth = info[2].as<IntImm>()->value;
        auto is_sender = info[3].as<IntImm>()->value;

        // create shared channel buffer 
        VarExpr channel_buf;
        if (channel_map_.count(channel)) {
          channel_buf = VarExpr(channel_map_[channel].node_);
        } else {
          channel_buf = VarExpr("c_buf_" + 
                            std::to_string(channel));
          channel_map_[channel] = channel_buf;
        }
        VarExpr target = VarExpr(op->args[pos].node_);
        auto shape = op->arg_shapes[pos];
          
        body = KernelRebuild(channel_buf, depth, channel, 
                   is_sender, target, shape, body);
      }
    }

    if (arg_scope_map_.count(op->name)) {
      auto set = arg_scope_map_[op->name];

      // insert annotation (pos : index = -1) indicate global
      for (size_t i = 0; i < op->args.size(); i++) {
        if (set.find(i) != set.end()) {
          // position, channel index and depth
          Array<Expr> info_new;
          info_new.push_back(IntImm::make(Int(32), i));
          info_new.push_back(IntImm::make(Int(32), -1));
          info_new.push_back(IntImm::make(Int(32), -1));
          info_new.push_back(IntImm::make(Int(32), -1));
          info_new.push_back(IntImm::make(Int(32), -1));
          info_new.push_back(IntImm::make(Int(32), -1));
          channels.push_back(info_new);
        }
      }
    }
    return KernelDef::make(
               op->args, op->arg_shapes, op->arg_types, 
               op->arg_tensors, body, op->ret_void, 
               op->ret_type, op->name, channels);
  }

  // attach atributes to kernel function calls 
  Stmt Mutate_(const KernelStmt* op, const Stmt& s) final {
    if (op->name == "test") {
      int count = 0;
      Array<Expr> keys, values;
      for (auto& arg : op->args) {
        auto name = arg.as<Variable>()->name_hint;
        // skip inner loop movement case 
        if (!mem_ports_.count(name)) {
          HCL_DEBUG(2) << "device function within loop or zerocopy mode";
          break;
        }
        auto dev_port = mem_ports_[name];
        CHECK(dev_port.size() == 4);
        // pos, channel index, depth, is_sedner, dev_type, mem_port
        keys.push_back(StringImm::make("pos"));
        values.push_back(IntImm::make(Int(32), count));

        keys.push_back(StringImm::make("mem"));
        values.push_back(dev_port[0]);
        keys.push_back(StringImm::make("port"));
        values.push_back(dev_port[1]);
        keys.push_back(StringImm::make("stream_type"));
        values.push_back(dev_port[2]);
        keys.push_back(StringImm::make("direction"));
        values.push_back(dev_port[3]);

        count = count + 1;
      }
      return KernelStmt::make(op->args, op->name, keys, values);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  unordered_map<string, unordered_set<int>> arg_scope_map_;
  unordered_map<int, VarExpr> channel_map_; 
  unordered_map<string, Array<Expr>> mem_ports_;
  unordered_set<const Variable*>& unused_vars_;

  // mutate kernel def body
  Stmt KernelRebuild(const VarExpr& channel_buf,
                     const int depth,
                     const int index,
                     const int is_sender,
                     const VarExpr& target_buf,
                     const Array<Expr> shape,
                     const Stmt& body) { 
    
    auto c_name = channel_buf.get()->name_hint;
    auto range_ = CollectIterRange(body);
    AccessCollector ac(target_buf, shape, range_, c_name); 
    ac.Mutate(body); 

    Stmt stmt;
    string target = target_buf.get()->name_hint;

    // self feedback loop
    if (is_sender == -1) {

    // sender mutate target store
    } else if (is_sender == 1) {
      if (ac.reg_store && ac.store_num == 1) {
        StoreToStreamStmtConverter mutator(
            target, StreamType::FIFO,
            channel_buf, depth, index, shape, range_); 
        stmt = mutator.Mutate(body);

      } else if (ac.store_num > 0) {
        if (!ac.reg_store)
          LOG(CLEAN) << "irregular \"" << target
                     << "\" access found; "
                     << "create reuse local buffer";
        if (ac.store_num > 1)
          LOG(CLEAN) << "multiple \"" << target
                     << "\" store found; "
                     << "create reuse local buffer";

        CHECK(ac.store_var.as<Variable>()) << "not a variable";
        VarExpr buf_var(ac.store_var.node_);
        stmt = BufferInserter(
                   body, shape, buf_var, channel_buf, false,
                   StreamType::FIFO, depth);
      } else {
        LOG(FATAL) << "target variable " 
                   << target << " not found; "
                   << "schedule does not apply";
      }

    // receiver mutate target load 
    } else if (is_sender == 0) {

      if (ac.reg_load && ac.load_num == 1) {
        LoadToStreamExprConverter mutator(
            target, StreamType::FIFO, 
            channel_buf, depth, index, shape, range_);
        stmt = mutator.Mutate(body);

      } else if (ac.load_num > 0) {
        if (!ac.reg_load)
          LOG(CLEAN) << "irregular \"" << target
                     << "\" access found; "
                     << "create reuse local buffer";
        if (ac.load_num > 1)
          LOG(CLEAN) << "multiple \"" << target
                     << "\" store found; "
                     << "create reuse local buffer";
        CHECK(ac.load_var.as<Variable>()) << "not a variable";
        VarExpr buf_var(ac.load_var.node_);
        stmt = BufferInserter(
                   body, shape, buf_var, channel_buf, true,
                   StreamType::FIFO, depth);
      } else {
        LOG(FATAL) << "target variable " 
                   << target << " not found; "
                   << "schedule does not apply";
      }
    }

    // create channel buffer
    if (not ac.buf_alloc) {
      auto dtype = channel_buf->type;
      stmt = Allocate::make(
                 VarExpr(channel_buf.node_), dtype, shape,
                 make_const(Bool(dtype.lanes()), true), stmt);
      stmt = AttrStmt::make(
                 VarExpr(channel_buf.node_), 
                 attr::storage_scope, 
                 StringImm::make("local"), stmt);
    }

    CHECK(stmt.defined());
    return stmt;
  }
};


Stmt InferStream(Stmt stmt, Array<NodeRef> api_args) {

  // Parse the IO interface information
  StreamInfoCollector sic(api_args);
  stmt = sic.Mutate(stmt);

  // If any inter-stage or inter-module varibles, 
  // 1. insert StreamStmt into its attr scope
  // 2. Create streaming channels (explicitly) for inter-stage 
  AllocateAttrDecorator aad(sic.global_channel_trace,
                            sic.inter_stage_channels, sic.dtype_, sic.shape_);
  stmt = aad.Mutate(stmt);

  // Mutate the device_scope AttrStmt 
  KernelDefCreator kdc(sic.dev_io_info, sic.shape_, sic.dtype_);
  stmt = kdc.SplitScope(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
