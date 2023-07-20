# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class HIRType:
    def __init__(self, bits):
        pass

    def __getitem__(self, shape):
        pass


bool = HIRType(1)
int1 = HIRType(1)
int8 = HIRType(8)
int16 = HIRType(16)
int32 = HIRType(32)
float16 = HIRType(16)
float32 = HIRType(32)
float64 = HIRType(64)
