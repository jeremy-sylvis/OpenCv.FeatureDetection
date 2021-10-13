using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using OpenCv.FeatureDetection.ImageProcessing;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCv.FeatureDetection.Console
{
    /// <summary>
    /// Result of feature detection.
    /// </summary>
    public class FeatureDetectionResult
    {
        public string FileName { get; private set; }

        public MKeyPoint[] KeyPoints { get; private set; }

        public int TotalFeatureCount { get; private set; }
        public int InlierFeatureCount { get; private set; }
        public float InlierOutlierRatio { get { return (float)InlierFeatureCount / (float)TotalFeatureCount; } }

        public int ExecutionTimeMs { get; private set; }

        public string FeatureDetector { get; private set; }
        public string FeatureDetectorConfiguration { get; private set; }

        public FeatureDetectionResult(string fileName, MKeyPoint[] keyPoints, int totalFeatureCount, int inlierFeatureCount, int executionTimeMs, string featureDetector, string featureDetectorConfiguration)
        {
            FileName = fileName;
            KeyPoints = keyPoints;
            TotalFeatureCount = totalFeatureCount;
            InlierFeatureCount = inlierFeatureCount;
            ExecutionTimeMs = executionTimeMs;
            FeatureDetector = featureDetector;
            FeatureDetectorConfiguration = featureDetectorConfiguration;
        }
    }

    /// <summary>
    /// An image to process through feature detection fuzzing.
    /// </summary>
    public class ImageToProcess
    {
        public string FileName { get; set; }
        public Rectangle RegionOfInterest { get; set; }
    }

    /// <summary>
    /// Helper class assisting in deserialization of <see cref="Rectangle"/>
    /// </summary>
    public class RectangleConverter : CustomCreationConverter<Rectangle>
    {
        public override bool CanWrite => false;
        public override bool CanRead => true;

        public override Rectangle Create(Type objectType)
        {
            return new Rectangle();
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            var jsonObject = JObject.Load(reader);

            var x = jsonObject["X"].Value<int>();
            var y = jsonObject["Y"].Value<int>();
            var height = jsonObject["Height"].Value<int>();
            var width = jsonObject["Width"].Value<int>();

            var rectangle = new Rectangle(new Point(x, y), new Size(width, height));
            return rectangle;
        }
    }

    /// <summary>
    /// A fuzzer for detecting the effective algorithms (and parameter sets) for detecting keypoints in the defined region of interest for an image.
    /// </summary>
    public class FeatureDetectorFuzzer
    {
        public const string InputFileName = "fuzzer-input.json";
        public const string OutputFileName = "fuzzer-output.csv";

        private readonly FuzzFeatureDetectorParameters _parameters;
        private readonly Logger _logger;
        private readonly ImageDrawing _imageDrawing;

        public FeatureDetectorFuzzer(FuzzFeatureDetectorParameters parameters, Logger logger, ImageDrawing imageDrawing)
        {
            _parameters = parameters;
            _logger = logger;
            _imageDrawing = imageDrawing;
        }

        /// <summary>
        /// Get our input images, run them through the various permutations of feature detectors and parameters, and prepare the output report and output images.
        /// </summary>
        public async Task FuzzFeatureDetectors()
        {
            var imagesToProcess = GetInputImages();

            // TODO: Ensure output is cleared before run
            var outputFilePath = Path.Combine(_parameters.OutputPath, OutputFileName);
            using (var outputFile = File.Create(outputFilePath))
            using (var outputStreamWriter = new StreamWriter(outputFile))
            {
                foreach (var imageToProcess in imagesToProcess)
                {
                    var imagePath = Path.Combine(_parameters.InputPath, imageToProcess.FileName);
                    if (!File.Exists(imagePath))
                    {
                        _logger.WriteMessage($"Error: Could not find file for image {imageToProcess.FileName}");
                        continue;
                    }

                    _logger.WriteMessage($"Processing file {imageToProcess.FileName}");
                    using (var imageUmat = new UMat(imagePath, Emgu.CV.CvEnum.ImreadModes.Color))
                    {
                        var akazeResults = await FuzzAkaze(imageToProcess, imageUmat);
                        await WriteDetectionResults(outputStreamWriter, akazeResults, imageUmat, imageToProcess.RegionOfInterest);
                    }
                }
            }
        }

        /// <summary>
        /// Write the results of feature detection to outputs.
        /// </summary>
        /// <param name="featureDetectionResults"></param>
        /// <returns></returns>
        private async Task WriteDetectionResults(StreamWriter outputStream, IList<FeatureDetectionResult> featureDetectionResults, UMat sourceImage, Rectangle regionOfInterest)
        {
            var outputFilePath = Path.Combine(_parameters.OutputPath, OutputFileName);

            for (var index = 0; index < featureDetectionResults.Count; index++)
            {
                var featureDetectionResult = featureDetectionResults[index];
                var outputFileName = $"{featureDetectionResult.FileName.Remove(featureDetectionResult.FileName.Length - 4, 4)}-{featureDetectionResult.FeatureDetector}-{index}.jpg";

                await Task.Run(() =>
                {
                    // Clone the output image for drawing
                    using (var outputImage = sourceImage.Clone())
                    {
                        _imageDrawing.DrawKeypointsOn(outputImage, featureDetectionResult.KeyPoints);
                        _imageDrawing.DrawRectangle(outputImage, regionOfInterest);
                        outputImage.Save(outputFileName);
                    }
                });

                var csvMessage = string.Join(',', featureDetectionResult.FileName, featureDetectionResult.InlierFeatureCount, featureDetectionResult.TotalFeatureCount,
                    featureDetectionResult.InlierOutlierRatio, featureDetectionResult.ExecutionTimeMs, featureDetectionResult.FeatureDetector, featureDetectionResult.FeatureDetectorConfiguration);
                await outputStream.WriteLineAsync(csvMessage);
            }
        }

        /// <summary>
        /// Fuzz parameters to the <see cref="AKAZE"/> feature detector.
        /// 
        /// Some helpful notes regarding parameters:
        /// descriptorSize: measured in bits; 0 is "full". KAZE uses 64 and 128-bit descriptors. We're going to specifically avoid fuzzing this.
        /// descriptorChannels: "channels" per descriptor
        /// threshold: Affects, somehow, quantity of keypoints detected. Default is 0.001. We'll fuzz that through 0.051 in 0.005 increments.
        /// octaves: entirely unknown. Default is 4. We'll fuzz 1-6.
        /// octaveLevels: entirely unknown. Default is 4. We'll fuzz 1-6.
        /// 
        /// Available documentation regarding AKAZE has proven exceedingly poor.
        /// 
        /// Sources:
        /// https://github.com/opencv/opencv/blob/master/modules/features2d/src/kaze/AKAZEConfig.h#L56
        /// https://answers.opencv.org/question/198674/what-is-the-relationship-between-akaze-descriptor-size-and-the-akaze-feature-patch-size/
        /// </summary>
        /// <param name="imageToProcess"></param>
        /// <param name="image"></param>
        /// <returns></returns>
        private async Task<IList<FeatureDetectionResult>> FuzzAkaze(ImageToProcess imageToProcess, UMat image)
        {
            var results = new List<FeatureDetectionResult>();
            var stopwatch = new System.Diagnostics.Stopwatch();

            // TODO: We should probably allow people to opt-in to invariant transform options
            var descriptorTypes = new[] { AKAZE.DescriptorType.KazeUpright, AKAZE.DescriptorType.MldbUpright };

            foreach (var descriptorType in descriptorTypes)
            {
                foreach (KAZE.Diffusivity diffusivityType in (KAZE.Diffusivity[])Enum.GetValues(typeof(KAZE.Diffusivity)))
                {
                    for (float threshold = 0.001f; threshold < 0.051f; threshold += 0.005f)
                    {
                        for (var octaves = 1; octaves <= 6; octaves++)
                        {
                            for (var octaveLevels = 1; octaveLevels <= 6; octaveLevels++)
                            {
                                await Task.Run(() =>
                                {
                                    using (var featureDetector = new AKAZE(descriptorType: descriptorType, threshold: threshold, nOctaves: octaves, nOctaveLayers: octaveLevels, diffusivity: diffusivityType))
                                    {
                                        stopwatch.Start();

                                        var keypoints = featureDetector.Detect(image);

                                        stopwatch.Stop();
                                        stopwatch.Reset();

                                        // Set results
                                        var keypointsInRegionOfInterest = keypoints.Count(x => IsPointInRegionOfInterest(x.Point, imageToProcess.RegionOfInterest));
                                        var parameters = $"\"descriptorType: {descriptorType}, diffusivityType: {diffusivityType}, threshold: {threshold}, octaves: {octaves}, octaveLevels: {octaveLevels}\"";
                                        var result = new FeatureDetectionResult(imageToProcess.FileName, keypoints, keypoints.Length, keypointsInRegionOfInterest, 1, "AKAZE", parameters);
                                        results.Add(result);
                                    }
                                });
                            }
                        }
                    }
                }
            }

            return results;
        }

        private static bool IsPointInRegionOfInterest(PointF point, Rectangle regionOfInterest)
        {
            return regionOfInterest.Left <= point.X && point.X <= regionOfInterest.Right &&
                regionOfInterest.Bottom <= point.Y && point.Y <= regionOfInterest.Top;
        }

        /// <summary>
        /// Scan the input path for files related to fuzzing feature detectors.
        /// 
        /// Expected:
        /// * config file describing:
        ///     * filenames of images to detect
        ///     * Region of Interest in that file to monitor for "positive" features
        /// * files in which to detect featuers
        /// </summary>
        private IEnumerable<ImageToProcess> GetInputImages()
        {
            var filePath = Path.Combine(_parameters.InputPath, InputFileName);

            if (!File.Exists(filePath))
            {
                throw new InvalidOperationException($"Fuzzer input file {InputFileName} not found at input path {_parameters.InputPath}");
            }

            string fileText;
            using (var file = File.Open(filePath, FileMode.Open))
            using (var streamReader = new StreamReader(file))
            {
                fileText = streamReader.ReadToEnd();
            }

            var imagesToProcess = JsonConvert.DeserializeObject<IEnumerable<ImageToProcess>>(fileText, new RectangleConverter());
            return imagesToProcess;
        }
    }
}
