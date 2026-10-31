namespace LucidTensor.Core;

/// <summary>Controls graph recording for the current asynchronous execution flow.</summary>
public static class Autograd
{
    private static readonly AsyncLocal<int> DisabledDepth = new();
    public static bool IsRecording => DisabledDepth.Value == 0;

    public static IDisposable NoGrad()
    {
        DisabledDepth.Value++;
        return new Scope();
    }

    private sealed class Scope : IDisposable
    {
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
                return;

            DisabledDepth.Value--;
            _disposed = true;
        }
    }
}
