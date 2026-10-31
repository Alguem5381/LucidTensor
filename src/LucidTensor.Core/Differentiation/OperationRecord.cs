namespace LucidTensor.Core;

/// <summary>One forward execution, including constant operands needed by its derivative.</summary>
public sealed class OperationRecord
{
    private Func<Tensor, Tensor[]>? _propagateGradient;
    private readonly int[] _versions;

    public string Name { get; }
    public IReadOnlyList<Tensor> Inputs { get; private set; }
    public bool IsReleased => _propagateGradient is null;

    internal OperationRecord(string name, Tensor[] inputs, Func<Tensor, Tensor[]> propagateGradient)
    {
        Name = name;
        Inputs = Array.AsReadOnly(inputs);

        _versions = [.. inputs.Select(input => input.Version)];
        _propagateGradient = propagateGradient;
    }

    internal void Validate()
    {
        if (IsReleased)
            throw new InvalidOperationException(
                "This graph was released. Run a new forward or use retainGraph: " +
                "true on the first backward.");

        if (Inputs.Where((tensor, index) => tensor.Version != _versions[index]).Any())
        {
            throw new InvalidOperationException(
                $"An input saved by {Name} was modified after forward. " +
                $"Finish backward before updating parameters.");
        }
    }

    internal Tensor[] PropagateGradient(Tensor outputGradient)
        => _propagateGradient!(outputGradient);

    internal void Release()
    {
        _propagateGradient = null;
        Inputs = [];
    }
}
