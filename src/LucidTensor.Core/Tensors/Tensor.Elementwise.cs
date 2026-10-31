namespace LucidTensor.Core;

public sealed partial class Tensor
{
    private static Tensor Binary(
        Tensor left,
        Tensor right,
        string name,
        Func<float, float, float> forward,
        Func<Tensor, Tensor[]> propagateGradient)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        var leftIsScalar = left.Rank == 0;
        var rightIsScalar = right.Rank == 0;
        var shapesMatch = left.Shape.SequenceEqual(right.Shape);

        if (!shapesMatch && !leftIsScalar && !rightIsScalar)
            throw new ArgumentException("Elementwise shapes must match, except for rank-zero scalars.");

        // O escalar acompanha o shape do outro operando.
        var outputShape = left._shape;

        if (leftIsScalar)
            outputShape = right._shape;


        var outputLength = Math.Max(left.Length, right.Length);
        var outputData = new float[outputLength];

        for (var index = 0; index < outputLength; index++)
        {
            // Um escalar reutiliza seu único elemento em todas as posições.
            var leftIndex = leftIsScalar ? 0 : index;
            var rightIndex = rightIsScalar ? 0 : index;

            var leftValue = left._data[leftIndex];
            var rightValue = right._data[rightIndex];

            outputData[index] = forward(leftValue, rightValue);
        }

        var output = new Tensor(outputShape, outputData);

        return Record(output, name, [left, right], PropagateToInputs);

        Tensor[] PropagateToInputs(Tensor outputGradient)
        {
            var contributions = propagateGradient(outputGradient);

            var leftGradient = ReduceGradientToInput(contributions[0], left);
            var rightGradient = ReduceGradientToInput(contributions[1], right);

            return [leftGradient, rightGradient];
        }
    }

    private static Tensor ReduceGradientToInput(Tensor gradient, Tensor input)
    {
        if (input.Rank == 0 && gradient.Rank != 0)
            return gradient.Sum();

        return gradient;
    }

    public static Tensor operator +(Tensor a, Tensor b)
        => Binary(
            left: a,
            right: b,
            name: "Add",
            forward: (x, y) => x + y,
            propagateGradient: g => [g, g]);

    public static Tensor operator -(Tensor a, Tensor b)
        => Binary(
            left: a,
            right: b,
            name: "Subtract",
            forward: (x, y) => x - y,
            propagateGradient: g => [g, -g]);

    public static Tensor operator /(Tensor a, Tensor b)
        => Binary(
            left: a,
            right: b,
            name: "Divide",
            forward: (x, y) => x / y,
            propagateGradient: g => [g / b, -(g.MultiplyElements(a) / b.Square())]);

    public Tensor MultiplyElements(Tensor other)
        => Binary(
            left: this,
            right: other,
            name: "MultiplyElements",
            forward: (x, y) => x * y,
            propagateGradient: g => [g.MultiplyElements(other), g.MultiplyElements(this)]);

    public Tensor Square() => MultiplyElements(this);

    public static Tensor operator +(Tensor a, float b) => a + Scalar(b);
    public static Tensor operator +(float a, Tensor b) => Scalar(a) + b;

    public static Tensor operator -(Tensor a, float b) => a - Scalar(b);
    public static Tensor operator -(float a, Tensor b) => Scalar(a) - b;

    public static Tensor operator *(Tensor a, float b) => a.MultiplyElements(Scalar(b));
    public static Tensor operator *(float a, Tensor b) => b * a;

    public static Tensor operator /(Tensor a, float b) => a / Scalar(b);
    public static Tensor operator /(float a, Tensor b) => Scalar(a) / b;

    public static Tensor operator -(Tensor a) => a * -1f;
}
