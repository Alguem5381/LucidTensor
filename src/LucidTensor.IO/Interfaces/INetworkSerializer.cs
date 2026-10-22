namespace LucidTensor.IO.Interfaces;
using LucidTensor.IO.Models;


public interface INetworkSerializer
{
    Task SaveAsync(NetworkState network, string filePath);
    Task<NetworkState?> LoadAsync(string filePath);
    Task<bool> IsValidAsync(string filePath);
}