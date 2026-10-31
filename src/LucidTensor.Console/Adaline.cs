using LucidTensor.Core;

internal sealed class Adaline : Module
{
    public Tensor Weight { get; }
    public Tensor Bias { get; }

    public Adaline()
    {
        Weight = RegisterParameter(nameof(Weight), new Tensor([0.1f, 0.2f]));
        Bias = RegisterParameter(nameof(Bias), Tensor.Scalar(0));
    }

    public Tensor Forward(Tensor input)
        => Weight.Dot(input) + Bias;
}
