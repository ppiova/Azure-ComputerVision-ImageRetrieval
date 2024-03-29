# 🚀 Azure Computer Vision Project 🌐

This project leverages Azure Computer Vision API to vectorize images and text, then computes the cosine similarity between the resulting vectors.

## 📋 Requirements

- .NET 5.0 or higher
- .NET libraries: Newtonsoft.Json, System.Net.Http, ConsoleTables
- Azure account and subscription key for Azure Computer Vision API

## 🏃‍♂️ Usage

First, you need to set up your Azure `endpoint` and `subscriptionKey` as environment variables:


private static string endpoint = Environment.GetEnvironmentVariable("COMPUTER_VISION_ENDPOINT");
private static string subscriptionKey = Environment.GetEnvironmentVariable("COMPUTER_VISION_SUBSCRIPTION_KEY");

Then, simply run the program and you'll see a table of similarities between the specified image and a set of text prompts in the console.

## 🛠️ Functions
The program performs the following operations:

- **ImageEmbedding(string imageUrl):** Takes an image URL and returns the vector of the image using Azure Computer Vision API.
- **TextEmbedding(string text):** Takes a text and returns the vector of the text using Azure Computer Vision API.
- **CosineSimilarity(double[] vector1, double[] vector2):** Computes and returns the cosine similarity between two vectors.
- **SimilarityResults(double[] imageVector, string[] prompts):** Computes the similarity between the image and several texts, and returns the results sorted by similarity.

The main program (Main) uses these functions to compute and display the similarity between an image and several text prompts.

## 
![Example image](https://github.com/ppiova/Azure-ComputerVision-ImageRetrieval/blob/master/readmeImages/imageExample.png "image")
![Example image](https://github.com/ppiova/Azure-ComputerVision-ImageRetrieval/blob/master/readmeImages/TablePrompt-Similarity.png "Table")

## Additional Documentation to learn more 📚

To learn more about Azure Computer Vision used in this project, please refer to the official Microsoft Learn documentation:

- [Microsoft Learn Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-image-retrieval?WT.mc_id=AI-MVP-5004753)
- [Create computer vision solutions with Azure Cognitive Services](https://bit.ly/computervisionsolutionswithAzure
)

## 👥 Contributions
Contributions are welcome. Feel free to open an Issue or make a Pull Request.

## 📄 License
This project is licensed under the terms of the MIT license.

Please ensure to replace the environment variables `"COMPUTER_VISION_ENDPOINT"` and `"COMPUTER_VISION_SUBSCRIPTION_KEY"` with your actual Azure data.
