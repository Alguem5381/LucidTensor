namespace LucidTensor.Core;

public sealed partial class Tensor
{
    internal int Version { get; private set; }

    public bool RequiresGrad { get; private set; }
    public bool IsLeaf => Creator is null;

    public OperationRecord? Creator { get; private set; }

    /// <summary>Accumulated gradient for leaves. Intermediate gradients are local to Backward.</summary>
    public Tensor? Grad { get; private set; }

    public Tensor RequireGradient()
    {
        if (!IsLeaf)
            throw new InvalidOperationException("Only leaves can be registered as parameters.");

        RequiresGrad = true;
        return this;
    }

    public Tensor Detach() => new(_shape, _data);

    public void ZeroGrad() => Grad = null;

    /// <summary>Explicit update of a leaf, permitted only inside NoGrad.</summary>
    public void AddInPlace(Tensor other, float alpha = 1f)
    {
        ValidateUpdate(other, alpha);

        for (var i = 0; i < Length; i++)
            _data[i] += alpha * other._data[i];

        Version++;
    }

    internal void ValidateUpdate(Tensor other, float alpha)
    {
        ArgumentNullException.ThrowIfNull(other);

        if (Autograd.IsRecording || !IsLeaf)
            throw new InvalidOperationException("In-place updates require a leaf tensor " +
                                                "and an Autograd.NoGrad scope.");

        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Update shapes must match.", nameof(other));

        if (!float.IsFinite(alpha))
            throw new ArgumentOutOfRangeException(nameof(alpha));
    }

    private static Tensor Record(
        Tensor result,
        string name,
        Tensor[] inputs,
        Func<Tensor, Tensor[]> propagateGradient)
    {
        if (!Autograd.IsRecording || !inputs.Any(input => input.RequiresGrad))
            return result;

        result.RequiresGrad = true;
        result.Creator = new OperationRecord(name, inputs, propagateGradient);

        return result;
    }

    /// <summary>Reverse-mode differentiation. Leaf gradients accumulate until ZeroGrad.</summary>
    public void Backward(Tensor? gradient = null, bool retainGraph = false)
    {
        if (!RequiresGrad)
            throw new InvalidOperationException("This tensor does not track gradients.");

        if (gradient is null && Rank != 0)
            throw new InvalidOperationException("A non-scalar output requires an explicit gradient seed.");

        // A derivada de uma perda escalar por ela mesma é 1.
        // Para saída não escalar, o chamador fornece uma semente com o mesmo shape.
        gradient ??= Scalar(1f);

        if (!Shape.SequenceEqual(gradient.Shape))
            throw new ArgumentException("Gradient seed shape must match output.", nameof(gradient));

        // O percurso entrega cada entrada antes dos resultados que dependem dela.
        // Validamos todo o histórico antes de começar a calcular os gradientes.
        var order = new List<Tensor>();

        foreach (var value in TraversePostOrder(this))
        {
            value.Creator?.Validate();
            order.Add(value);
        }

        // As contas do backward não devem criar outro grafo de diferenciação
        using var noGrad = Autograd.NoGrad();

        // gradients[t] representa a derivada da saída escolhida em relação a t.
        // Os gradientes intermediários existem apenas durante esta chamada.
        var gradients = new Dictionary<Tensor, Tensor> { [this] = gradient.Detach() };

        // Invertemos a lista para percorrer dos resultados até as entradas. Assim,
        // todas as contribuições chegam a um tensor antes de ele propagar seu gradiente.
        order.Reverse();

        foreach (var value in order)
        {
            var g = gradients[value];

            if (value.Creator is not { } creator)
                continue;

            // Cada operação aplica sua regra da cadeia e devolve uma contribuição
            var contributions = creator.PropagateGradient(g);

            // Em u + u, as duas entradas são o mesmo objeto, logo somamos as duas
            // contribuições. Visitar u uma só vez não elimina suas ocorrências em uma soma.
            for (var index = 0; index < creator.Inputs.Count; index++)
            {
                var input = creator.Inputs[index];

                if (!input.RequiresGrad)
                    continue;

                gradients[input] = gradients.TryGetValue(input, out var previous)
                    ? previous + contributions[index]
                    : contributions[index];
            }
        }

        // Só as folhas conservam Grad. Chamadas diferentes também acumulam nele
        foreach (var value in order.Where(value => value.IsLeaf))
            value.Grad = value.Grad is null
                ? gradients[value]
                : value.Grad + gradients[value];

        // Liberamos o que foi salvo para derivação
        if (retainGraph)
            return;

        foreach (var value in order)
            value.Creator?.Release();
    }

    private enum TraversalAction
    {
        VisitInputs,
        YieldTensor
    }

    // DFS em pós-ordem equivale a visitar as entradas recursivamente
    private static IEnumerable<Tensor> TraversePostOrder(Tensor root)
    {
        var seen = new HashSet<Tensor>();
        var stack = new Stack<(Tensor Value, TraversalAction Action)>();

        stack.Push((root, TraversalAction.VisitInputs));

        while (stack.TryPop(out var entry))
        {
            if (entry.Action == TraversalAction.YieldTensor)
            {
                // Entregamos o tensor ao foreach.
                yield return entry.Value;
                continue;
            }

            // Um tensor compartilhado é entregue uma única vez.
            if (!seen.Add(entry.Value))
                continue;

            // O lembrete entra primeiro e as entradas empilhadas depois, já que vão sair antes
            stack.Push((entry.Value, TraversalAction.YieldTensor));

            if (entry.Value.Creator is not { } creator)
                continue;

            foreach (var input in creator.Inputs)
                if (input.RequiresGrad)
                    stack.Push((input, TraversalAction.VisitInputs));
        }
    }
}
