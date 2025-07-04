# fixed_point_utils.py
import numpy as np
from myhdl import intbv
# Q format: Q_I.F (I integer bits, F fractional bits)
# Let's choose Q16.16 for 32-bit fixed point
FIXED_POINT_F_BITS = 16 # Fractional bits
FIXED_POINT_SCALE = 2**FIXED_POINT_F_BITS
FIXED_POINT_TOTAL_BITS = 32 # Total bits (e.g., 32-bit signed integer)

# Assuming these are your core fixed-point definitions (e.g., 32-bit, 16 fractional)
FIXED_POINT_TOTAL_BITS = 32
FIXED_POINT_F_BITS = 16
FIXED_POINT_I_BITS = FIXED_POINT_TOTAL_BITS - FIXED_POINT_F_BITS # Integer bits (including sign)

# --- NEW CONSTANTS FOR WIDER ARITHMETIC ---
# Product result width:
# A fixed-point number with Total_Bits and F_Bits has (Total_Bits - F_Bits) integer bits.
# When two such numbers are multiplied, the integer part can grow up to 2 * (Total_Bits - F_Bits).
# The fractional part becomes 2 * F_Bits.
# After scaling (right shift by F_Bits), the resulting fixed-point number (product) will have:
# Integer bits: 2 * FIXED_POINT_I_BITS
# Fractional bits: FIXED_POINT_F_BITS
# Total bits for the product: (2 * FIXED_POINT_I_BITS) + FIXED_POINT_F_BITS
PRODUCT_ACC_BITS = (2 * FIXED_POINT_I_BITS) + FIXED_POINT_F_BITS # For 32-bit, 16-frac: (2*16) + 16 = 32 + 16 = 48 bits

# Accumulator width:
# The accumulator sums up 'matmul_A_cols' products.
# If matmul_A_cols can be up to 256 (from SHAPE_DIM_BITS=8), then we need log2(256) = 8 extra bits
# to prevent overflow from the sum of 256 product terms.
ACCUMULATOR_EXTRA_BITS = 8 # Sufficient for up to 2^8 = 256 accumulations
ACCUMULATOR_TOTAL_BITS = PRODUCT_ACC_BITS + ACCUMULATOR_EXTRA_BITS # 48 + 8 = 56 bits


def float_to_fixed(value):
    """Converts a float to its fixed-point integer representation (Q16.16)."""
    max_representable = (2**(FIXED_POINT_TOTAL_BITS - FIXED_POINT_F_BITS -1)) - (1 / FIXED_POINT_SCALE)
    min_representable = -(2**(FIXED_POINT_TOTAL_BITS - FIXED_POINT_F_BITS -1))
    clamped_value = np.clip(value, min_representable, max_representable)
    fixed_value = int(clamped_value * FIXED_POINT_SCALE)
    return fixed_value

def fixed_to_float(fixed_value):
    """Converts a fixed-point integer representation (Q16.16) back to float."""
    return float(fixed_value) / FIXED_POINT_SCALE
# fixed_point_utils.py

# Ensure these constants are correctly defined and match your desired hardware bit-widths
FIXED_POINT_F_BITS = 16 # Example: 16 fractional bits
FIXED_POINT_TOTAL_BITS = 32 # Example: 32 total bits (1 sign, 15 integer, 16 fractional)
# So, DATA_WIDTH in tpu_core.py should be 32

# For addition, result might need 1 more bit temporarily for sum before clamping
# For multiplication, raw product needs 2 * FIXED_POINT_TOTAL_BITS
PRODUCT_RAW_BITS = FIXED_POINT_TOTAL_BITS * 2
# Accumulator needs to be wide enough for the sum of K products.
# Let's say K_MAX is the max number of elements in K loop (e.g., if max col/row is 256 (SHAPE_DIM_BITS=8), then K_MAX is 256)
# ACCUMULATOR_TOTAL_BITS = PRODUCT_RAW_BITS + (SHAPE_DIM_BITS or ceil(log2(K_MAX)))
# Assuming SHAPE_DIM_BITS for matrix dimensions is 8, max K = 2^8 = 256. log2(256) = 8.
# ACCUMULATOR_TOTAL_BITS = (FIXED_POINT_TOTAL_BITS * 2) + 8 # Example
ACCUMULATOR_TOTAL_BITS = 32 + 8 # Example, or whatever you determined

# Helper to clamp / saturate an intbv to a specific bit_width
def _clamp_intbv(value, total_bits):
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))
    
    if value > max_val:
        return intbv(max_val)[total_bits:]
    elif value < min_val:
        return intbv(min_val)[total_bits:]
    else:
        return intbv(value)[total_bits:]

def fixed_add(a, b):
    # Perform the sum, letting MyHDL's intbv handle intermediate width
    sum_val = a.signed() + b.signed()
    # Clamp the result to FIXED_POINT_TOTAL_BITS (which should be DATA_WIDTH)
    return _clamp_intbv(sum_val, FIXED_POINT_TOTAL_BITS)

def fixed_mul(a, b):
    # Raw multiplication
    prod_raw = a.signed() * b.signed()
    
    # Scale back to the desired fixed-point format (e.g., shift right by fractional bits)
    # This might implicitly handle the sign extension correctly for negative numbers.
    scaled_prod = prod_raw >> FIXED_POINT_F_BITS 
    
    # The output of fixed_mul, when directly used in `fixed_add(acc, product)`,
    # should typically be either `FIXED_POINT_TOTAL_BITS` or `PRODUCT_ACC_BITS` depending
    # on if it's a "standard product" or one meant for direct accumulation.
    # Given `matmul_mac_acc = fixed_add(matmul_mac_acc, product)` where `product` is this output,
    # `product` likely needs to be `ACCUMULATOR_TOTAL_BITS` wide, or `fixed_add` must handle the width difference.
    # For now, let's assume it should match `FIXED_POINT_TOTAL_BITS` for intermediate calculation
    # and `fixed_add` will manage the accumulator width.
    return _clamp_intbv(scaled_prod, FIXED_POINT_TOTAL_BITS) # Or a wider width if this is `PRODUCT_ACC_BITS`

def fixed_relu(val):
    # Relu should also return an intbv of the correct width
    # `val.signed()` gives the signed integer value for comparison
    return _clamp_intbv(max(0, val.signed()), FIXED_POINT_TOTAL_BITS) # Ensure output matches DATA_WIDTH
