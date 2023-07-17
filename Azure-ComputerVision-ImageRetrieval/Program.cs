using ConsoleTables;
using Newtonsoft.Json;
using System.Text;

public class AzureCV
{
    private static readonly HttpClient client = new HttpClient();

    private static readonly string endpoint = Environment.GetEnvironmentVariable("COMPUTER_VISION_ENDPOINT");
    private static string version = "?api-version=2023-02-01-preview&modelVersion=latest";
    private static string vecImgUrl = endpoint + "/retrieval:vectorizeImage" + version;
    private static string vecTxtUrl = endpoint + "/retrieval:vectorizeText" + version;

    private static readonly string subscriptionKey = Environment.GetEnvironmentVariable("COMPUTER_VISION_SUBSCRIPTION_KEY");

    private static async Task<float[]> ImageEmbedding(string imageUrl)
    {
        var image = new { url = imageUrl };
        var content = new StringContent(JsonConvert.SerializeObject(image), Encoding.UTF8, "application/json");
        client.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", subscriptionKey);

        var response = await client.PostAsync(vecImgUrl, content);
        var responseJson = await response.Content.ReadAsStringAsync();
        dynamic responseObj = JsonConvert.DeserializeObject<dynamic>(responseJson);
        var vector = responseObj.vector.ToObject<float[]>();

        return vector;
    }

    private static async Task<float[]> TextEmbedding(string text)
    {
        var prompt = new { text = text };
        var content = new StringContent(JsonConvert.SerializeObject(prompt), Encoding.UTF8, "application/json");
        client.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", subscriptionKey);

        var response = await client.PostAsync(vecTxtUrl, content);
        var responseJson = await response.Content.ReadAsStringAsync();
        var responseObj = JsonConvert.DeserializeObject<dynamic>(responseJson);
        var vector = responseObj.vector.ToObject<float[]>();

        return vector;
    }
    
    private static float CosineSimilarity(float[] vector1, float[] vector2)
    {
        float dotProduct = 0;
        int length = Math.Min(vector1.Length, vector2.Length);
        for (int i = 0; i< length; i++)
        {
            dotProduct += vector1[i] * vector2[i];
        }
        float magnitude1 = (float)Math.Sqrt(vector1.Select(x=> x*x).Sum());
        float magnitude2 = (float)Math.Sqrt(vector2.Select(x => x * x).Sum());

        //var dotProduct = vector1.Zip(vector2, (a, b) => a * b).Sum();

        //var magnitude1 = Math.Sqrt(vector1.Sum(x => x * x));
        //var magnitude2 = Math.Sqrt(vector2.Sum(x => x * x));

        return dotProduct / (magnitude1 * magnitude2);
    }

    private static async Task<List<Tuple<string, float>>> SimilarityResults(float[] imageVector, string[] prompts)
    {
        var results = new List<Tuple<string, float>>();
        foreach (var prompt in prompts)
        {
            var textVector = await TextEmbedding(prompt);
            var similarity = CosineSimilarity(imageVector, textVector);
            results.Add(new Tuple<string, float>(prompt, similarity));
        }

        var sortedResults = results.OrderByDescending(x => x.Item2).ToList();

        return sortedResults;
    }

    public static async Task Main(string[] args)
    {
        string imageUrl = "https://raw.githubusercontent.com/ppiova/Azure-ComputerVision-BackgroundRemoval/main/Images/01F56FEF-8415-4851-AB94-14321F8FBE28.jpg";
        float[] imageVector = await ImageEmbedding(imageUrl);

        string[] prompts = new string[]
        {
            "dog", "a truck", "a car", "a green car", "a white car",
            "a Toyota SUV white car", "a tesla car", "a ford car", "a motorbike", "a renault car"
        };
        var sortedResults = await SimilarityResults(imageVector, prompts);

        var table = new ConsoleTable("Prompt", "Similarity");
        foreach (var result in sortedResults)
        {
            table.AddRow(result.Item1, result.Item2);
        }
        table.Write();
    }
}
