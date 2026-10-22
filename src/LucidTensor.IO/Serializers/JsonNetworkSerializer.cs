namespace LucidTensor.IO.Serializers;

using System.Text.Json;
using LucidTensor.IO.Interfaces;
using LucidTensor.IO.Models;

public class JsonNetworkSerializer : INetworkSerializer
{
    //reserva um local no sistema de arquivo do pc e salva lá um json contando info da rede
    public async Task SaveAsync(NetworkState network, string filePath)
    {
        await using var stream = File.Create(filePath);
        await JsonSerializer.SerializeAsync(stream, network, new JsonSerializerOptions { WriteIndented = true });
    }
    
    //simplesmente carrega um objeto json em um determinado local do sistema.
    public async Task<NetworkState?> LoadAsync(string filePath)
    {
        
        if (!File.Exists(filePath)) return null;
        await using var stream = File.OpenRead(filePath);
        return await JsonSerializer.DeserializeAsync<NetworkState>(stream);
    }
    
     //verifica se o json é válido
    public async Task<bool> IsValidAsync(string filePath)
    {
        if (!File.Exists(filePath)) return false;

        try
        {
            //se o usuário escolher um json qualquer que não siga estrutura do NetworkState, lança exceção
            await using var stream =  File.OpenRead(filePath);
            var state = await JsonSerializer.DeserializeAsync<NetworkState>(stream);
            //verifica se o arquivo tem pelo menos uma camada da ree neural
            return state != null && state.Layers != null & state.Layers.Count > 0;
        }
        catch(JsonException)
        {
            
            return false;
        }
        

    }
    
}