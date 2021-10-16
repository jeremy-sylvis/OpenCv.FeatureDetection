using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace OpenCv.FeatureDetection.Console
{
    public class AgastRunner
    {
        public IList<AgastParameters> GetParameters(ImageToProcess imageParameters, Mat image)
        {
            var parameters = new List<AgastParameters>();

            var agastTypes = new[] { AgastFeatureDetector.Type.AGAST_5_8, AgastFeatureDetector.Type.AGAST_7_12d, AgastFeatureDetector.Type.AGAST_7_12s, AgastFeatureDetector.Type.OAST_9_16 };
            foreach (var agastType in agastTypes)
            {
                for (int threshold = 2; threshold < 20f; threshold += 2)
                {
                    foreach (var useNonMaxSuppression in new[] { true, false })
                    {
                        parameters.Add(new AgastParameters(imageParameters, image, threshold, useNonMaxSuppression, agastType));
                    }
                }
            }

            return parameters;
        }

        public FeatureDetectionResult PerformDetection(AgastParameters parameters)
        {
            using (var featureDetector = new AgastFeatureDetector(threshold: parameters.Threshold, nonmaxSuppression: parameters.UseNonMaxSuppression, type: parameters.AgastType))
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
                var parameterText = $"\"agastType: {parameters.AgastType}, threshold: {parameters.Threshold}, useNonMaxSuppression: {parameters.UseNonMaxSuppression}\"";
                var result = new FeatureDetectionResult(parameters.ImageParameters.FileName, keypoints, keypoints.Length, keypointsInRegionOfInterest, stopwatch.ElapsedMilliseconds, "AGAST", parameterText);

                return result;
            }
        }

        private static bool IsPointInRegionOfInterest(PointF point, Rectangle regionOfInterest)
        {
            // Note: With how this counts, bottom is the _higher_ value
            return regionOfInterest.Left <= point.X && point.X <= regionOfInterest.Right &&
                regionOfInterest.Top <= point.Y && point.Y <= regionOfInterest.Bottom;
        }
    }

    /// <summary>
    /// Parameters describing an AKAZE feature detector/extractor.
    /// </summary>
    public struct AgastParameters
    {
        public ImageToProcess ImageParameters { get; private set; }
        public Mat Image { get; private set; }

        public int Threshold { get; private set; }
        public bool UseNonMaxSuppression { get; private set; }
        public AgastFeatureDetector.Type AgastType { get; private set; }

        public AgastParameters(ImageToProcess imageParameters, Mat image, int threshold, bool useNonMaxSuppression, AgastFeatureDetector.Type agastType)
        {
            ImageParameters = imageParameters;
            Image = image;

            Threshold = threshold;
            UseNonMaxSuppression = useNonMaxSuppression;
            AgastType = agastType;
        }
    }
}
