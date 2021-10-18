using Emgu.CV;
using Emgu.CV.XFeatures2D;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Linq;

namespace OpenCv.FeatureDetection.Console
{
    public class StarRunner : FeatureDetectorRunner<StarParameters>
    {
        public override IList<StarParameters> GetParameters(ImageToProcess imageParameters, Mat image)
        {
            var parameters = new List<StarParameters>();

            // TODO: We should probably allow people to opt-in to invariant transform options
            for (var maxSize = 25; maxSize <= 65; maxSize += 5)
            {
                for (var responseThreshold = 10; responseThreshold <= 50; responseThreshold += 10)
                {
                    for (var lineThreshold = 4; lineThreshold <= 16; lineThreshold += 2)
                    {
                        for (var lineThresholdBinarized = 4; lineThresholdBinarized <= 14; lineThresholdBinarized += 2)
                        {
                            for (var suppressNonMaxSize = 1; suppressNonMaxSize <= 15; suppressNonMaxSize += 2)
                            {
                                parameters.Add(new StarParameters(imageParameters, image, maxSize, responseThreshold, lineThreshold, lineThresholdBinarized, suppressNonMaxSize));
                            }
                        }
                        
                    }
                }
            }

            return parameters;
        }

        public override FeatureDetectionResult PerformDetection(StarParameters parameters)
        {
            using (var featureDetector = new StarDetector(parameters.MaxSize, parameters.ResponseThreshold, parameters.LineThresholdProjected, parameters.LineThresholdBinarized, parameters.SuppressNonMaxSize))
            {
                var stopwatch = new System.Diagnostics.Stopwatch();
                stopwatch.Start();

                MKeyPoint[] keypoints = null;
                using (var imageUmat = parameters.Image.GetUMat(Emgu.CV.CvEnum.AccessType.Read))
                {
                    keypoints = featureDetector.Detect(parameters.Image);
                }

                stopwatch.Stop();

                // Set results
                var keypointsInRegionOfInterest = keypoints.Count(x => IsPointInRegionOfInterest(x.Point, parameters.ImageParameters.RegionOfInterest));
                var parameterText = $"\"maxSize: {parameters.MaxSize}, responseThreshold: {parameters.ResponseThreshold}, lineThresholdProjected: {parameters.LineThresholdProjected}, lineThresholdBinarized: {parameters.LineThresholdBinarized}, suppressNonMaxSize: {parameters.SuppressNonMaxSize}\"";
                var result = new FeatureDetectionResult(parameters.ImageParameters.FileName, keypoints, keypoints.Length, keypointsInRegionOfInterest, stopwatch.ElapsedMilliseconds, "STAR", parameterText);

                return result;
            }
        }
    }

    /// <summary>
    /// Parameters describing an STAR feature detector/extractor.
    /// </summary>
    public struct StarParameters
    {
        public ImageToProcess ImageParameters { get; private set; }
        public Mat Image { get; private set; }

        public int MaxSize { get; private set; }
        public int ResponseThreshold { get; private set; }
        public int LineThresholdProjected { get; private set; }
        public int LineThresholdBinarized { get; private set; }
        public int SuppressNonMaxSize { get; private set; }

        public StarParameters(ImageToProcess imageParameters, Mat image, int maxSize, int responseThreshold, int lineThresholdProjected, int lineThresholdBinarized, int suppressNonMaxSize)
        {
            ImageParameters = imageParameters;
            Image = image;

            MaxSize = maxSize;
            ResponseThreshold = responseThreshold;
            LineThresholdProjected = lineThresholdProjected;
            LineThresholdBinarized = lineThresholdBinarized;
            SuppressNonMaxSize = suppressNonMaxSize;
        }
    }
}
