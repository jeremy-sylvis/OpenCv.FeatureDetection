using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCv.FeatureDetection.Console
{
    public class AkazeRunner
    {
        public IList<AkazeParameters> GetParameters(ImageToProcess imageParameters, Mat image)
        {
            var akazeParameters = new List<AkazeParameters>();

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
                                akazeParameters.Add(new AkazeParameters(imageParameters, image, descriptorType, threshold, octaves, octaveLevels, diffusivityType));
                            }
                        }
                    }
                }
            }

            return akazeParameters;
        }

        public FeatureDetectionResult PerformDetection(AkazeParameters parameters)
        {
            using (var featureDetector = new AKAZE(descriptorType: parameters.DescriptorType, threshold: parameters.Threshold, nOctaves: parameters.Octaves, nOctaveLayers: parameters.OctaveLayers, diffusivity: parameters.DiffusivityType))
            {
                var stopwatch = new System.Diagnostics.Stopwatch();
                stopwatch.Start();

                MKeyPoint[] keypoints = null;
                using (var imageUmat = parameters.Image.GetUMat(Emgu.CV.CvEnum.AccessType.Read)) 
                {
                    keypoints = featureDetector.Detect(parameters.Image);
                }
                
                stopwatch.Stop();
                stopwatch.Reset();

                // Set results
                var keypointsInRegionOfInterest = keypoints.Count(x => IsPointInRegionOfInterest(x.Point, parameters.ImageParameters.RegionOfInterest));
                var parameterText = $"\"descriptorType: {parameters.DescriptorType}, diffusivityType: {parameters.DiffusivityType}, threshold: {parameters.Threshold}, octaves: {parameters.Octaves}, octaveLayers: {parameters.OctaveLayers}\"";
                var result = new FeatureDetectionResult(parameters.ImageParameters.FileName, keypoints, keypoints.Length, keypointsInRegionOfInterest, stopwatch.ElapsedMilliseconds, "AKAZE", parameterText);
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
    public struct AkazeParameters
    {
        public ImageToProcess ImageParameters { get; private set; }
        public Mat Image { get; private set; }
        public AKAZE.DescriptorType DescriptorType { get; private set; }
        public float Threshold { get; private set; }
        public int Octaves { get; private set; }
        public int OctaveLayers { get; private set; }
        public KAZE.Diffusivity DiffusivityType { get; private set; }

        public AkazeParameters(ImageToProcess imageParameters, Mat image, AKAZE.DescriptorType descriptorType, float threshold, int octaves, int octaveLayers, KAZE.Diffusivity diffusivityType)
        {
            ImageParameters = imageParameters;
            Image = image;
            DescriptorType = descriptorType;
            Threshold = threshold;
            Octaves = octaves;
            OctaveLayers = octaveLayers;
            DiffusivityType = diffusivityType;
        }
    }
}
