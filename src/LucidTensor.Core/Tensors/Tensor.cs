using System.Globalization;

namespace LucidTensor.Core;

/// <summary>FP32 tensor. Arithmetic creates new values; explicit leaf updates preserve identity.</summary>
public sealed partial class Tensor
{
    private readonly int[] _shape;
    private readonly float[] _data;

    public int Rank => _shape.Length;
    public int Length => _data.Length;

    public ReadOnlySpan<int> Shape => _shape;
    public ReadOnlySpan<float> Data => _data;

    public Tensor(float[] data, bool requiresGrad = false)
        : this([data?.Length ?? throw new ArgumentNullException(nameof(data))], data, requiresGrad)
    {
    }

    public Tensor(Array data, bool requiresGrad = false)
        : this(InferShape(data), [.. data.Cast<float>()], requiresGrad)
    {
    }

    public Tensor(int[] shape, float[] data, bool requiresGrad = false)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(data);

        _shape = [.. shape];

        if (_shape.Any(d => d <= 0))
            throw new ArgumentException("Shape dimensions must be positive.", nameof(shape));

        int expected;

        try
        {
            expected = _shape.Aggregate(1, (product, d) => checked(product * d));
        }
        catch (OverflowException)
        {
            throw new ArgumentException("Shape is too large.", nameof(shape));
        }

        if (expected != data.Length)
            throw new ArgumentException($"Shape requires {expected} elements, received {data.Length}.", nameof(data));

        _data = [.. data];

        RequiresGrad = requiresGrad;
    }

    private static int[] InferShape(Array data)
    {
        ArgumentNullException.ThrowIfNull(data);

        return data.GetType().GetElementType() != typeof(float)
            ? throw new ArgumentException("Expected a rectangular float array.", nameof(data))
            : [.. Enumerable.Range(0, data.Rank).Select(data.GetLength)];
    }

    public static Tensor Scalar(float value, bool requiresGrad = false)
        => new([], [value], requiresGrad);

    public float Item() => Length == 1
        ? _data[0]
        : throw new InvalidOperationException("Item requires exactly one element.");

    [System.Runtime.CompilerServices.IndexerName("Element")]
    public float this[params int[] indexes]
    {
        get
        {
            ArgumentNullException.ThrowIfNull(indexes);

            if (indexes.Length != Rank)
                throw new ArgumentException("Provide one index per dimension.", nameof(indexes));

            var offset = 0;

            for (var d = 0; d < Rank; d++)
            {
                if (indexes[d] < 0 || indexes[d] >= _shape[d])
                    throw new ArgumentOutOfRangeException(nameof(indexes));

                offset = offset * _shape[d] + indexes[d];
            }

            return _data[offset];
        }
    }

    public override string ToString() =>
        $"Tensor([{string.Join(", ", _shape)}], [{string.Join(", ", _data.Select(v => v.ToString("G6", CultureInfo.InvariantCulture)))}])";
}
