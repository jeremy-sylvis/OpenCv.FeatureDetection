using Emgu.CV;
using System.Collections.Generic;
using System.Drawing;

namespace OpenCv.FeatureDetection.Console
{
    public abstract class FeatureDetectorRunner<TParam>
    {
        public abstract IList<TParam> GetParameters(ImageToProcess imageParameters, Mat image);
        public abstract FeatureDetectionResult PerformDetection(TParam parameters);

        protected bool IsPointInRegionOfInterest(PointF point, Rectangle regionOfInterest)
        {
            // Note: With how this counts, bottom is the _higher_ value
            return regionOfInterest.Left <= point.X && point.X <= regionOfInterest.Right &&
                regionOfInterest.Top <= point.Y && point.Y <= regionOfInterest.Bottom;
        }
    }
}
