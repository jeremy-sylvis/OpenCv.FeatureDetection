using System;
using System.Threading.Tasks;
using OpenCv.FeatureDetection.Console.Data;
using OpenCv.FeatureDetection.ImageProcessing;

namespace OpenCv.FeatureDetection.Console
{
    class Program
    {
        static void Main(string[] args)
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
            var siftRunner = new SiftRunner();

            var databasePath = System.IO.Path.Combine(parameters.FuzzFeatureDetectorParameters.OutputPath, "FeatureDetectorFuzzerResults.sqlite");

            switch (parameters.Operation)
            {
                case ParameterParser.FuzzFeatureDetectorsOperation:
                    using (var dbContext = new ConsoleDbContext(databasePath))
                    {
                        dbContext.Database.EnsureCreated();
                        dbContext.SaveChanges();

                        System.Console.WriteLine("Fuzzing feature detectors. Warning: This may take a while to complete.");
                        var featureDetectorFuzzer = new FeatureDetectorFuzzer(parameters.FuzzFeatureDetectorParameters, logger, imageDrawing, akazeRunner, agastRunner, orbRunner, starRunner, siftRunner, dbContext);
                        featureDetectorFuzzer.FuzzFeatureDetectors();

                        dbContext.SaveChanges();
                    }

                    break;
                default:
                    throw new Exception($"Could not process operation {parameters.Operation}");
            }

            System.Console.WriteLine("Press any key to exit...");
            System.Console.ReadKey();
        }
    }
}
