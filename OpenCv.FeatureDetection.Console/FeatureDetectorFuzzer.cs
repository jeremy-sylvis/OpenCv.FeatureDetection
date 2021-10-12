using System;
using System.Collections.Generic;
using System.Drawing;
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
        public Rectangle RegionOfInterest { get; private set; }
    }

    /// <summary>
    /// A fuzzer for detecting the effective algorithms (and parameter sets) for detecting keypoints in the defined region of interest for an image.
    /// </summary>
    public class FeatureDetectorFuzzer
    {
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
            throw new NotImplementedException();
        }
    }
}
