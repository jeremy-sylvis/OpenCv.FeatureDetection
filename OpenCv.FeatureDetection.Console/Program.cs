using System;

namespace OpenCv.FeatureDetection.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            System.Console.WriteLine("OpenCv.FeatureDetection.Console");

            var parameterParser = new ParameterParser();
            var parameters = parameterParser.Parse(args);

            switch (parameters.Operation)
            {
                case ParameterParser.FuzzFeatureDetectorsOperation:
                    System.Console.WriteLine("Fuzzing feature detectors. Warning: This may take a while to complete.");
                    var featureDetectorFuzzer = new FeatureDetectorFuzzer(parameters.FuzzFeatureDetectorParameters);
                    featureDetectorFuzzer.FuzzFeatureDetectors();
                    break;
                default:
                    throw new Exception($"Could not process operation {parameters.Operation}");
            }

            System.Console.WriteLine("Press any key to exit...");
            System.Console.ReadKey();
        }
    }
}
