-- This query converts a bytea column, which stores a packed array of single-precision floats,
-- into a human-readable JSON array. This version uses a pure SQL function and has no
-- external dependencies like Python.

-- Step 1: Create a pure SQL helper function to convert 4 bytes into a float.
-- This function manually reconstructs an IEEE 754 single-precision float from its
-- raw bytes using bitwise operations. It only needs to be created once.
-- It assumes the bytea is stored in little-endian format, which is standard for
-- numpy.tobytes() on most modern hardware (like x86 and ARM).
CREATE OR REPLACE FUNCTION public.bytea_to_float4_sql(bytes bytea)
RETURNS real AS $$
DECLARE
    -- The 4 bytes converted to a 32-bit integer.
    int_val int;
    -- The components of an IEEE 754 single-precision float.
    sign int;
    exponent int;
    mantissa int;
BEGIN
    -- Ensure the input is exactly 4 bytes long.
    IF length(bytes) != 4 THEN
        RAISE EXCEPTION 'Input bytea must be 4 bytes long to be converted to a float4. Length was %', length(bytes);
    END IF;

    -- Convert the 4 little-endian bytes into a single 32-bit integer.
    int_val := (get_byte(bytes, 3) << 24) | (get_byte(bytes, 2) << 16) | (get_byte(bytes, 1) << 8) | get_byte(bytes, 0);

    -- Extract the sign bit (the most significant bit).
    sign := (int_val >> 31) & 1;

    -- Extract the 8-bit exponent.
    exponent := (int_val >> 23) & 255;

    -- Extract the 23-bit mantissa (the fractional part).
    mantissa := int_val & 8388607; -- 8388607 is (2^23 - 1)

    -- Handle special cases based on the IEEE 754 standard.
    IF exponent = 255 THEN
        IF mantissa = 0 THEN
            -- Infinity
            RETURN CASE WHEN sign = 0 THEN 'Infinity'::real ELSE '-Infinity'::real END;
        ELSE
            -- Not a Number (NaN)
            RETURN 'NaN'::real;
        END IF;
    END IF;

    IF exponent = 0 THEN
        IF mantissa = 0 THEN
            -- Zero
            RETURN 0.0;
        ELSE
            -- Denormalized number
            RETURN power(-1, sign) * (mantissa / power(2, 23)) * power(2, -126);
        END IF;
    END IF;

    -- Reconstruct the float value for standard normalized numbers.
    RETURN power(-1, sign) * (1 + mantissa / power(2, 23)) * power(2, exponent - 127);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Step 2: Use the new function in your query.
-- This query now correctly unnests and converts the bytea embedding.
WITH unnested_floats AS (
  SELECT
    e.item_id,
    -- Use our new pure SQL function to convert each 4-byte chunk.
    public.bytea_to_float4_sql(substring(e.embedding FROM s.byte_offset + 1 FOR 4)) AS val,
    s.byte_offset
  FROM
    -- Subquery to limit the number of items being processed for performance.
    (SELECT item_id, embedding FROM public.embedding ORDER BY item_id LIMIT 100) AS e
  -- Create a row for the starting position of each 4-byte float.
  CROSS JOIN LATERAL generate_series(0, length(e.embedding) - 4, 4) AS s(byte_offset)
)
-- Final aggregation to build the JSON array for each item.
SELECT
  item_id,
  to_jsonb(array_agg(val ORDER BY byte_offset)) AS embedding_json
FROM unnested_floats
GROUP BY item_id;
