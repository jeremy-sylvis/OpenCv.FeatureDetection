using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.XFeatures2D;
using Emgu.CV.Structure;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using OpenCv.FeatureDetection.ImageProcessing;
using OpenCv.FeatureDetection.ImageProcessing.Extensions;
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
        public int OutlierFeatureCount { get { return TotalFeatureCount - OutlierFeatureCount; } }
        public float InlierOutlierRatio { get { return (float)InlierFeatureCount / (float)TotalFeatureCount; } }

        public long ExecutionTimeMs { get; private set; }

        public string FeatureDetector { get; private set; }
        public string FeatureDetectorConfiguration { get; private set; }

        public FeatureDetectionResult(string fileName, MKeyPoint[] keyPoints, int totalFeatureCount, int inlierFeatureCount, long executionTimeMs, string featureDetector, string featureDetectorConfiguration)
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
        
        private readonly AkazeRunner _akazeRunner;
        private readonly AgastRunner _agastRunner;
        private readonly OrbRunner _orbRunner;
        private readonly StarRunner _starRunner;
        private readonly SiftRunner _siftRunner;

        public FeatureDetectorFuzzer(FuzzFeatureDetectorParameters parameters, Logger logger, ImageDrawing imageDrawing, AkazeRunner akazeRunner, AgastRunner agastRunner, OrbRunner orbRunner, StarRunner starRunner, SiftRunner siftRunner)
        {
            _parameters = parameters;
            _logger = logger;
            _imageDrawing = imageDrawing;
            _akazeRunner = akazeRunner;
            _agastRunner = agastRunner;
            _orbRunner = orbRunner;
            _starRunner = starRunner;
            _siftRunner = siftRunner;
        }

        /// <summary>
        /// Get our input images, run them through the various permutations of feature detectors and parameters, and prepare the output report and output images.
        /// </summary>
        public async Task FuzzFeatureDetectors()
        {
            var imagesToProcess = GetInputImages();

            // TODO: Ensure output is cleared before run
            var reportFilePath = Path.Combine(_parameters.OutputPath, OutputFileName);
            using (var reportfile = File.Create(reportFilePath))
            using (var reportStreamWriter = new StreamWriter(reportfile))
            {
                reportStreamWriter.AutoFlush = true;

                // Insert a header row
                var headerRow = "FileName,Inliers,Total,Inlier/Outlier Ratio,Execution (ms),Algorithm,Parameters";
                reportStreamWriter.WriteLine(headerRow);

                foreach (var imageToProcess in imagesToProcess)
                {
                    var imagePath = Path.Combine(_parameters.InputPath, imageToProcess.FileName);
                    if (!File.Exists(imagePath))
                    {
                        _logger.WriteMessage($"Error: Could not find file for image {imageToProcess.FileName}");
                        continue;
                    }

                    _logger.WriteMessage($"Processing file {imageToProcess.FileName}");

                    using (var imageMat = new Mat(imagePath, Emgu.CV.CvEnum.ImreadModes.Color))
                    {
                        var akazeResults = FuzzAkaze(imageToProcess, imageMat);
                        Parallel.ForEach(akazeResults, (x, y, index) =>
                        {
                            WriteDetectionResults(reportStreamWriter, x, (int)index, imageMat, imageToProcess.RegionOfInterest);
                        });

                        var agastResults = FuzzAgast(imageToProcess, imageMat);
                        Parallel.ForEach(agastResults, (x, y, index) =>
                        {
                            WriteDetectionResults(reportStreamWriter, x, (int)index, imageMat, imageToProcess.RegionOfInterest);
                        });

                        var orbResults = FuzzOrb(imageToProcess, imageMat);
                        Parallel.ForEach(orbResults, (x, y, index) =>
                        {
                            WriteDetectionResults(reportStreamWriter, x, (int)index, imageMat, imageToProcess.RegionOfInterest);
                        });

                        var starResults = FuzzStar(imageToProcess, imageMat);
                        Parallel.ForEach(starResults, (x, y, index) =>
                        {
                            WriteDetectionResults(reportStreamWriter, x, (int)index, imageMat, imageToProcess.RegionOfInterest);
                        });

                        var siftResults = FuzzSift(imageToProcess, imageMat);
                        Parallel.ForEach(siftResults, (x, y, index) =>
                        {
                            WriteDetectionResults(reportStreamWriter, x, (int)index, imageMat, imageToProcess.RegionOfInterest);
                        });
                    }
                }
            }
        }

        /// <summary>
        /// Write the results of feature detection to outputs.
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        private void WriteDetectionResults(StreamWriter outputStream, FeatureDetectionResult result, int index, Mat sourceImage, Rectangle regionOfInterest)
        {
            var generatedImageFileName = $"{result.FileName.Remove(result.FileName.Length - 4, 4)}-{result.FeatureDetector}-{index}.jpg";
            var generatedImageFilePath = Path.Combine(_parameters.OutputPath, generatedImageFileName);

            // Clone the output image for drawing
            using (var outputImage = sourceImage.Clone())
            {
                _imageDrawing.DrawRectangleOn(outputImage, regionOfInterest);
                //using (var updatedOutputImage = _imageDrawing.DrawKeypoints(outputImage, result.KeyPoints))
                _imageDrawing.DrawKeypointsOn(outputImage, result.KeyPoints);
                {
                    outputImage.Save(generatedImageFilePath);
                }
            }

            var csvMessage = string.Join(',', generatedImageFileName, result.InlierFeatureCount, result.TotalFeatureCount, result.InlierOutlierRatio, result.ExecutionTimeMs, result.FeatureDetector, result.FeatureDetectorConfiguration);
            outputStream.WriteLine(csvMessage);
        }

        // TODO: These could *all* be condensed - the runner is all that varies
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
        private IEnumerable<FeatureDetectionResult> FuzzAkaze(ImageToProcess imageToProcess, Mat image)
        {
            var akazeRuns = _akazeRunner.GetParameters(imageToProcess, image);

            var skip = 0;
            var batchSize = 10;
            do
            {
                var batchResults = akazeRuns
                    .Skip(skip)
                    .Take(batchSize)
                    .AsParallel()
                    .Select(_akazeRunner.PerformDetection)
                    .ToArray();

                foreach (var batchResult in batchResults)
                {
                    yield return batchResult;
                }

                skip += batchSize;
                if (skip >= akazeRuns.Count) break;

            } while (true);
        }

        /// <summary>
        /// Fuzz parameters to the <see cref="AgastFeatureDetector"/> feature detector.
        /// 
        /// Available documentation regarding AGAST has proven exceedingly poor.
        /// 
        /// Sources:
        /// </summary>
        /// <param name="imageToProcess"></param>
        /// <param name="image"></param>
        /// <returns></returns>
        private IEnumerable<FeatureDetectionResult> FuzzAgast(ImageToProcess imageToProcess, Mat image)
        {
            var parameters = _agastRunner.GetParameters(imageToProcess, image);

            var skip = 0;
            var batchSize = 10;
            do
            {
                var batchResults = parameters
                    .Skip(skip)
                    .Take(batchSize)
                    .AsParallel()
                    .Select(_agastRunner.PerformDetection)
                    .ToArray();

                foreach (var batchResult in batchResults)
                {
                    yield return batchResult;
                }

                skip += batchSize;
                if (skip >= parameters.Count) break;

            } while (true);
        }

        /// <summary>
        /// Fuzz parameters to the <see cref="ORB"/> feature detector.
        /// 
        /// Available documentation regarding ORB has proven exceedingly poor.
        /// 
        /// Sources:
        /// </summary>
        /// <param name="imageToProcess"></param>
        /// <param name="image"></param>
        /// <returns></returns>
        private IEnumerable<FeatureDetectionResult> FuzzOrb(ImageToProcess imageToProcess, Mat image)
        {
            var parameters = _orbRunner.GetParameters(imageToProcess, image);

            var skip = 0;
            var batchSize = 10;
            do
            {
                var batchResults = parameters
                    .Skip(skip)
                    .Take(batchSize)
                    .AsParallel()
                    .Select(_orbRunner.PerformDetection)
                    .ToArray();

                foreach (var batchResult in batchResults)
                {
                    yield return batchResult;
                }

                skip += batchSize;
                if (skip >= parameters.Count) break;

            } while (true);
        }

        /// <summary>
        /// Fuzz parameters to the <see cref="StarDetector"/> feature detector.
        /// 
        /// Available documentation regarding STAR has proven exceedingly poor.
        /// 
        /// Sources:
        /// </summary>
        /// <param name="imageToProcess"></param>
        /// <param name="image"></param>
        /// <returns></returns>
        private IEnumerable<FeatureDetectionResult> FuzzStar(ImageToProcess imageToProcess, Mat image)
        {
            var parameters = _starRunner.GetParameters(imageToProcess, image);

            var skip = 0;
            var batchSize = 10;
            do
            {
                var batchResults = parameters
                    .Skip(skip)
                    .Take(batchSize)
                    .AsParallel()
                    .Select(_starRunner.PerformDetection)
                    .ToArray();

                foreach (var batchResult in batchResults)
                {
                    yield return batchResult;
                }

                skip += batchSize;
                if (skip >= parameters.Count) break;

            } while (true);
        }

        /// <summary>
        /// Fuzz parameters to the <see cref="SIFT"/> feature detector.
        /// 
        /// Available documentation regarding SIFT has proven exceedingly poor.
        /// 
        /// Sources:
        /// </summary>
        /// <param name="imageToProcess"></param>
        /// <param name="image"></param>
        /// <returns></returns>
        private IEnumerable<FeatureDetectionResult> FuzzSift(ImageToProcess imageToProcess, Mat image)
        {
            var parameters = _siftRunner.GetParameters(imageToProcess, image);

            var skip = 0;
            var batchSize = 10;
            do
            {
                var batchResults = parameters
                    .Skip(skip)
                    .Take(batchSize)
                    .AsParallel()
                    .Select(_siftRunner.PerformDetection)
                    .ToArray();

                foreach (var batchResult in batchResults)
                {
                    yield return batchResult;
                }

                skip += batchSize;
                if (skip >= parameters.Count) break;

            } while (true);
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
