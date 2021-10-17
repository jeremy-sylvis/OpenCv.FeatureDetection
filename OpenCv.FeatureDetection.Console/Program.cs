using System;
using System.Threading.Tasks;
using OpenCv.FeatureDetection.ImageProcessing;

namespace OpenCv.FeatureDetection.Console
{
    class Program
    {
        static async Task Main(string[] args)
        {
            System.Console.WriteLine("OpenCv.FeatureDetection.Console");

            var parameterParser = new ParameterParser();
            var parameters = parameterParser.Parse(args);
            var logger = new Logger();
            var imageDrawing = new ImageDrawing();
            var akazeRunner = new AkazeRunner();
            var agastRunner = new AgastRunner();
            var orbRunner = new OrbRunner();
            var starRunner = new StarRunner();

            switch (parameters.Operation)
            {
                case ParameterParser.FuzzFeatureDetectorsOperation:
                    System.Console.WriteLine("Fuzzing feature detectors. Warning: This may take a while to complete.");
                    var featureDetectorFuzzer = new FeatureDetectorFuzzer(parameters.FuzzFeatureDetectorParameters, logger, imageDrawing, akazeRunner, agastRunner, orbRunner, starRunner);
                    await featureDetectorFuzzer.FuzzFeatureDetectors();
                    break;
                default:
                    throw new Exception($"Could not process operation {parameters.Operation}");
            }

            System.Console.WriteLine("Press any key to exit...");
            System.Console.ReadKey();
        }
    }
}
