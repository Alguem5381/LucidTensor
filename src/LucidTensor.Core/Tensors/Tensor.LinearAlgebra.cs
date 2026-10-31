namespace LucidTensor.Core;

public sealed partial class Tensor
{
    /// <summary>Rank-2 matrix multiplication, or elementwise multiplication when either operand is rank zero.</summary>
    public static Tensor operator *(Tensor a, Tensor b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Rank == 0 || b.Rank == 0)
            return a.MultiplyElements(b);

        if (a.Rank != 2 || b.Rank != 2 || a._shape[1] != b._shape[0])
            throw new ArgumentException("Matrix multiplication requires [m,n] and [n,p].");

        var rows = a._shape[0];
        var inner = a._shape[1];
        var cols = b._shape[1];

        if ((long)rows * cols > int.MaxValue)
            throw new ArgumentException("Result is too large.");

        var data = new float[rows * cols];

        for (var row = 0; row < rows; row++)
        for (var col = 0; col < cols; col++)
        {
            double sum = 0;

            for (var k = 0; k < inner; k++)
                sum += (double)a[row, k] * b[k, col];

            data[row * cols + col] = (float)sum;
        }

        return Record(new Tensor([rows, cols], data), "MatMul", [a, b],
            g => [g * b.Transpose(), a.Transpose() * g]);
    }

    public Tensor Transpose()
    {
        if (Rank != 2)
            throw new InvalidOperationException("Transpose requires rank 2.");

        var rows = _shape[0];
        var cols = _shape[1];
        var data = new float[Length];

        for (var row = 0; row < rows; row++)
        for (var col = 0; col < cols; col++)
            data[col * rows + row] = this[row, col];

        return Record(new Tensor([cols, rows], data), "Transpose", [this], g => [g.Transpose()]);
    }

    public Tensor Dot(Tensor other)
    {
        ArgumentNullException.ThrowIfNull(other);

        if (Rank != 1 || other.Rank != 1 || Length != other.Length)
            throw new ArgumentException("Dot requires vectors of equal length.", nameof(other));

        double sum = 0;

        for (var i = 0; i < Length; i++)
            sum += (double)_data[i] * other._data[i];

        return Record(Scalar((float)sum), "Dot", [this, other],
            g => [other * g.Item(), this * g.Item()]);
    }

    public Tensor Sum()
    {
        double sum = _data.Aggregate<float, double>(0, (current, value) => current + value);

        return Record(Scalar((float)sum), "Sum", [this],
            g => [new Tensor(_shape, [.. Enumerable.Repeat(g.Item(), Length)])]);
    }

    public Tensor Mean()
        => Sum() / Length;
}
