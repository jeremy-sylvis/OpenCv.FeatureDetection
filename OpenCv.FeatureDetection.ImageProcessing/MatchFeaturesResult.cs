using Emgu.CV;
using Emgu.CV.Util;

namespace OpenCv.FeatureDetection.ImageProcessing
{
    public class MatchFeaturesResult
    {
        public VectorOfKeyPoint ObservedKeyPoints { get; private set; }
        public Mat ObservedDescriptors { get; set; }

        public VectorOfKeyPoint ModelKeyPoints { get; private set; }
        public Mat ModelDescriptors { get; set; }

        public VectorOfVectorOfDMatch Matches { get; set; }
        public Mat Mask { get; set; }
        public Mat Homography { get; set; }

        public MatchFeaturesResult(VectorOfKeyPoint observedKeyPoints, Mat observedDescriptors, VectorOfKeyPoint modelKeyPoints, Mat modelDescriptors, VectorOfVectorOfDMatch matches, Mat mask, Mat homography)
        {
            ObservedKeyPoints = observedKeyPoints;
            ObservedDescriptors = observedDescriptors;
            ModelKeyPoints = modelKeyPoints;
            ModelDescriptors = modelDescriptors;
            Matches = matches;
            Mask = mask;
            Homography = homography;
        }
    }
}
