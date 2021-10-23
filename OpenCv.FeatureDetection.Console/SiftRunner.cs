using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Linq;

namespace OpenCv.FeatureDetection.Console
{
    public class SiftRunner : FeatureDetectorRunner<SiftParameters>
    {
        public override IList<SiftParameters> GetParameters(ImageToProcess imageParameters, Mat image)
        {
            var akazeParameters = new List<SiftParameters>();

            //7*6*10*10*10
            //7*6*8*8*8
            // Note: 0 features is unlimited
            for (var features = 0; features < 1500; features += 250)
            {
                for (var octaveLayers = 1; octaveLayers <= 6; octaveLayers++)
                {
                    for (double contrastThreshold = 0.02d; contrastThreshold <= 0.09d; contrastThreshold += 0.01d)
                    {
                        for (double edgeThreshold = 4; edgeThreshold <= 18; edgeThreshold += 2)
                        {
                            for (double sigma = 1.2d; sigma <= 1.9d; sigma += 0.1d)
                            {
                                akazeParameters.Add(new SiftParameters(imageParameters, image, features, octaveLayers, contrastThreshold, edgeThreshold, sigma));
                            }
                        }
                    }
                }
            }

            return akazeParameters;
        }

        public override FeatureDetectionResult PerformDetection(SiftParameters parameters)
        {
            using (var featureDetector = new SIFT(parameters.Features, parameters.OctaveLayers, parameters.ContrastThreshold, parameters.EdgeThreshold, parameters.Sigma))
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
                var parameterText = $"\"features: {parameters.Features}, octaveLayers: {parameters.OctaveLayers}, contrastThreshold: {parameters.ContrastThreshold}, edgeThreshold: {parameters.EdgeThreshold}, sigma: {parameters.Sigma}\"";
                var result = new FeatureDetectionResult(parameters.ImageParameters.FileName, keypoints, keypoints.Length, keypointsInRegionOfInterest, stopwatch.ElapsedMilliseconds, "SIFT", parameterText);

                return result;
            }
        }
    }

    /// <summary>
    /// Parameters describing a SIFT feature detector/extractor.
    /// </summary>
    public struct SiftParameters
    {
        public ImageToProcess ImageParameters { get; private set; }
        public Mat Image { get; private set; }

        public int Features { get; private set; }
        public int OctaveLayers { get; private set; }
        public double ContrastThreshold { get; private set; }
        public double EdgeThreshold { get; private set; }
        public double Sigma { get; private set; }

        public SiftParameters(ImageToProcess imageParameters, Mat image, int features, int octaveLayers, double contrastThreshold, double edgeThreshold, double sigma)
        {
            ImageParameters = imageParameters;
            Image = image;

            Features = features;
            OctaveLayers = octaveLayers;
            ContrastThreshold = contrastThreshold;
            EdgeThreshold = edgeThreshold;
            Sigma = sigma;
        }
    }
}
