using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
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

        public int TotalFeatureCount { get; private set; }
        public int InlierFeatureCount { get; private set; }

        public string FeatureDetector { get; private set; }
        public string FeatureDetectorConfiguration { get; private set; }
    }

    /// <summary>
    /// An image to process through feature detection fuzzing.
    /// </summary>
    public class ImageToProcess
    {
        public string FileName { get; set; }
        public Rectangle RegionOfInterest { get; set; }
    }

    // Helper for deserializing Rectangle
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

        private readonly FuzzFeatureDetectorParameters _parameters;

        public FeatureDetectorFuzzer(FuzzFeatureDetectorParameters parameters)
        {
            _parameters = parameters;
        }

        /// <summary>
        /// Get our input images, run them through the various permutations of feature detectors and parameters, and prepare the output report and output images.
        /// </summary>
        public void FuzzFeatureDetectors()
        {
            var imagesToProcess = GetInputImages();
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
