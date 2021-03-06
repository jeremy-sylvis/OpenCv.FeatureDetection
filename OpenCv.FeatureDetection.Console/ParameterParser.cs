using System;
using System.Linq;

namespace OpenCv.FeatureDetection.Console
{
    /// <summary>
    /// The values of command-line parameters passed to this application.
    /// </summary>
    public class Parameters
    {
        public string Operation { get; private set; }

        public FuzzFeatureDetectorParameters FuzzFeatureDetectorParameters { get; private set; }

        public Parameters(string operation, FuzzFeatureDetectorParameters fuzzFeatureDetectorParameters)
        {
            Operation = operation;
            FuzzFeatureDetectorParameters = fuzzFeatureDetectorParameters;
        }
    }

    /// <summary>
    /// the values of command-line parameters relevant to feature detector fuzzing.
    /// </summary>
    public class FuzzFeatureDetectorParameters
    {
        public string InputPath { get; private set; }
        public string OutputPath { get; private set; }

        public FeatureDetectorAlgorithms? SpecifiedFeatureDetectorAlgorithms { get; private set; }

        public FuzzFeatureDetectorParameters(string inputPath, string outputPath, FeatureDetectorAlgorithms? featureDetectorAlgorithms)
        {
            InputPath = inputPath;
            OutputPath = outputPath;

            SpecifiedFeatureDetectorAlgorithms = featureDetectorAlgorithms;
        }
    }


    /// <summary>
    /// A basic parser for the recognized command-line parameters for this application.
    /// </summary>
    public class ParameterParser
    {
        private const string OperationParameter = "-Operation";

        public const string FuzzFeatureDetectorsOperation = "FuzzFeatureDetectors";

        public const string InputPathParameter = "-InputPath";
        public const string OutputPathParameter = "-OutputPath";
        public const string AlgorithmsParameter = "-Algorithms";

        /// <summary>
        /// Parse the given arguments for parameter information.
        /// </summary>
        /// <param name="arguments"></param>
        /// <returns></returns>
        public Parameters Parse(string[] arguments)
        {
            string operation = null;
            string[] remainingArguments = null;

            for (var index = 0; index < arguments.Length; index++)
            {
                if (arguments[index] == OperationParameter)
                {
                    // Operation requires a paired operand
                    if (index + 1 >= arguments.Length)
                    {
                        throw new Exception("'-Operation' requires a value. Accepted values: [ FuzzFeatureDetectors ]");
                    }

                    operation = arguments[index + 1];

                    // Skip up to our index, +1 for 1-based, +1 for value of -Operation
                    remainingArguments = arguments.Skip(index + 2).ToArray();
                    break;
                }
            }

            // FuzzFeatureDetectors is the only valid operation right now
            if (operation != FuzzFeatureDetectorsOperation)
            {
                throw new Exception("'-Operation' requires a value. Accepted values: [ FuzzFeatureDetectors ]");
            }

            var fuzzFeatureDetectorParameters = ParseFuzzFeatureDetectorsParameters(remainingArguments);
            
            var result = new Parameters(operation, fuzzFeatureDetectorParameters);
            return result;
        }

        /// <summary>
        /// Parse the given set of arguments for the InputPath and OutputPath parameters.
        /// </summary>
        /// <param name="arguments"></param>
        /// <returns></returns>
        private FuzzFeatureDetectorParameters ParseFuzzFeatureDetectorsParameters(string[] arguments)
        {
            string inputPath = null;
            string outputPath = null;

            FeatureDetectorAlgorithms? specifiedFeatureDetectorAlgorithms = null;

            for (var index = 0; index < arguments.Length; index++)
            {
                if (arguments[index] == InputPathParameter)
                {
                    // Operation requires a paired operand
                    if (index + 1 >= arguments.Length)
                    {
                        throw new Exception($"'{InputPathParameter}' requires a value.");
                    }

                    inputPath = arguments[index + 1];
                    index++;
                    continue;
                }

                // These are a good candidate for refactor / cleanup
                if (arguments[index] == OutputPathParameter)
                {
                    // Operation requires a paired operand
                    if (index + 1 >= arguments.Length)
                    {
                        throw new Exception($"'{OutputPathParameter}' requires a value.");
                    }

                    outputPath = arguments[index + 1];
                    index++;
                    continue;
                }

                if (arguments[index] == AlgorithmsParameter)
                {
                    //Operation requires a paired operand
                    if (index + 1 >= arguments.Length)
                    {
                        throw new Exception($"'{AlgorithmsParameter}' requires a value.");
                    }

                    // Algorithms is expected to be a CSV
                    var chosenAlgorithmsCsv = arguments[index + 1];
                    var chosenAlgorithms = chosenAlgorithmsCsv.Split(',');
                    foreach (var chosenAlgorithm in chosenAlgorithms)
                    {
                        // Account for any whitespace the user may have included
                        var cleanedChosenAlgorithm = chosenAlgorithm.Trim();

                        // Attempt to parse it as an enum value
                        FeatureDetectorAlgorithms featureDetectorAlgorithm = FeatureDetectorAlgorithms.AKAZE;
                        
                        if (!Enum.TryParse(cleanedChosenAlgorithm, true, out featureDetectorAlgorithm))
                        {
                            throw new ArgumentException($"Specified algorithm '{cleanedChosenAlgorithm}' was not recognized.");
                        }

                        // Set it in the result set
                        if (specifiedFeatureDetectorAlgorithms == null)
                        {
                            specifiedFeatureDetectorAlgorithms = featureDetectorAlgorithm;
                        }
                        else
                        {
                            specifiedFeatureDetectorAlgorithms |= featureDetectorAlgorithm;
                        }
                    }
                    
                    index++;
                    continue;
                }
            }

            if (inputPath == null)
            {
                throw new Exception("'-InputPath' is required.");
            }

            if (outputPath == null)
            {
                throw new Exception("'-OutputPath' is required.");
            }

            var result = new FuzzFeatureDetectorParameters(inputPath, outputPath, specifiedFeatureDetectorAlgorithms);
            return result;
        }
    }
}
