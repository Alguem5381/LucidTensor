namespace LucidTensor.Core;

/// <summary>A persistent registry of parameter tensors and child modules, independent of execution graphs.</summary>
public abstract class Module
{
    private readonly Dictionary<string, Tensor> _parameters = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Module> _modules = new(StringComparer.Ordinal);

    protected Tensor RegisterParameter(string name, Tensor tensor)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(tensor);

        if (_modules.ContainsKey(name))
            throw new InvalidOperationException($"{name} is already a submodule.");

        if (_parameters.TryGetValue(name, out var existing) && !ReferenceEquals(existing, tensor))
            throw new InvalidOperationException($"Parameter {name} was replaced. Update its " +
                                                $"data instead of replacing its reference.");

        tensor.RequireGradient();

        _parameters[name] = tensor;

        return tensor;
    }

    protected T RegisterModule<T>(string name, T module) where T : Module
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(module);

        if (_parameters.ContainsKey(name))
            throw new InvalidOperationException($"{name} is already a parameter.");

        if (_modules.TryGetValue(name, out var existing) && !ReferenceEquals(existing, module))
            throw new InvalidOperationException($"Submodule {name} was " +
                                                $"replaced after registration.");

        _modules[name] = module;

        return module;
    }

    public IReadOnlyList<Tensor> Parameters()
    {
        var result = new List<Tensor>();
        var tensors = new HashSet<Tensor>();

        var modules = new HashSet<Module>();
        var pending = new Stack<Module>();

        pending.Push(this);

        while (pending.TryPop(out var module))
        {
            if (!modules.Add(module))
                continue;

            result.AddRange(module._parameters.Values.Where(tensors.Add));

            foreach (var child in module._modules.Values)
                pending.Push(child);
        }

        return result.AsReadOnly();
    }
}
