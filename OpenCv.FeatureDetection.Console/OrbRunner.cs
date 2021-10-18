using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Linq;

namespace OpenCv.FeatureDetection.Console
{
    public class OrbRunner : FeatureDetectorRunner<OrbParameters>
    {
        public override IList<OrbParameters> GetParameters(ImageToProcess imageParameters, Mat image)
        {
            var parameters = new List<OrbParameters>();

            for (var numberOfFeatures = 250; numberOfFeatures <= 1500; numberOfFeatures += 250)
            {
                for (var scaleFactor = 1.1f; scaleFactor <= 1.4f; scaleFactor += 0.1f)
                {
                    for (var levels = 1; levels <= 4; levels++)
                    {
                        for (var edgeThreshold = 11; edgeThreshold <= 46; edgeThreshold += 5)
                        {
                            var scoreTypes = new[] { ORB.ScoreType.Fast, ORB.ScoreType.Harris };
                            foreach (var scoreType in scoreTypes)
                            {
                                for (var patchSize = 11; patchSize <= 46; patchSize += 5)
                                {
                                    for (var fastThreshold = 10; fastThreshold <= 30; fastThreshold += 5)
                                    {
                                        parameters.Add(new OrbParameters(imageParameters, image, numberOfFeatures, scaleFactor, levels, edgeThreshold, scoreType, patchSize, fastThreshold));
                                    }
                                }
                            }
                        }
                    }
                }
            }


            return parameters;
        }

        public override FeatureDetectionResult PerformDetection(OrbParameters parameters)
        {
            using (var featureDetector = new ORB(numberOfFeatures: parameters.NumberOfFeatures, scaleFactor: parameters.ScaleFactor, nLevels: parameters.Levels, edgeThreshold: parameters.EdgeThreshold, scoreType: parameters.ScoreType, patchSize: parameters.PatchSize, fastThreshold: parameters.FastThreshold))
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
                var parameterText = $"\"numberOfFeatures: {parameters.NumberOfFeatures}, scaleFactor: {parameters.ScaleFactor}, levels: {parameters.Levels}, edgeThreshold: {parameters.EdgeThreshold}, scoreType: {parameters.ScoreType}, patchSize: {parameters.PatchSize}, fastThreshold: {parameters.FastThreshold}\"";
                var result = new FeatureDetectionResult(parameters.ImageParameters.FileName, keypoints, keypoints.Length, keypointsInRegionOfInterest, stopwatch.ElapsedMilliseconds, "ORB", parameterText);

                return result;
            }
        }

    }

    /// <summary>
    /// Parameters describing an ORB feature detector/extractor.
    /// </summary>
    public struct OrbParameters
    {
        public ImageToProcess ImageParameters { get; private set; }
        public Mat Image { get; private set; }

        public int NumberOfFeatures { get; private set; }
        public float ScaleFactor { get; private set; }
        public int Levels { get; private set; }
        public int EdgeThreshold { get; private set; }
        public ORB.ScoreType ScoreType { get; private set; }
        public int PatchSize { get; private set; }
        public int FastThreshold { get; private set; }

        public OrbParameters(ImageToProcess imageParameters, Mat image, int numberOfFeatures, float scaleFactor, int levels, int edgeThreshold, ORB.ScoreType scoreType, int patchSize, int fastThreshold)
        {
            ImageParameters = imageParameters;
            Image = image;

            NumberOfFeatures = numberOfFeatures;
            ScaleFactor = scaleFactor;
            Levels = levels;
            EdgeThreshold = edgeThreshold;
            ScoreType = scoreType;
            PatchSize = patchSize;
            FastThreshold = fastThreshold;
        }
    }
}
