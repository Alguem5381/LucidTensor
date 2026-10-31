using LucidTensor.Core;

/// <summary>Otimizador de exemplo: consome gradientes usando apenas a API pública do núcleo.</summary>
internal sealed class Sgd
{
    private readonly Tensor[] _parameters;

    public float LearningRate { get; }

    public Sgd(IEnumerable<Tensor> parameters, float learningRate)
    {
        ArgumentNullException.ThrowIfNull(parameters);

        if (!float.IsFinite(learningRate) || learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate));

        _parameters = [.. parameters.Distinct()];

        foreach (var parameter in _parameters)
        {
            ArgumentNullException.ThrowIfNull(parameter);

            if (!parameter.IsLeaf || !parameter.RequiresGrad)
                throw new ArgumentException(
                    "Optimizer parameters must be gradient-tracking leaves.",
                    nameof(parameters)
                );
        }

        LearningRate = learningRate;
    }

    public void ZeroGrad()
    {
        foreach (var parameter in _parameters)
            parameter.ZeroGrad();
    }

    public void Step()
    {
        using var noGrad = Autograd.NoGrad();

        foreach (var parameter in _parameters)
            if (parameter.Grad is { } gradient)
                parameter.AddInPlace(gradient, -LearningRate);
    }
}
