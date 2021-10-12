using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace OpenCv.FeatureDetection.ImageProcessing
{
    public class ImageDrawing
    {
        /// <summary>
        /// Draw matches between the given model image and its keypoints and the given observed image and its keypoints using the given matches, mask of descriptors, and homography.
        /// </summary>
        /// <param name="modelImageMat"></param>
        /// <param name="modelKeyPoints"></param>
        /// <param name="observedImageMat"></param>
        /// <param name="observedKeyPoint"></param>
        /// <param name="matches"></param>
        /// <param name="mask"></param>
        /// <returns>Result image.</returns>
        public Mat DrawFeatureMatches(Mat modelImageMat, VectorOfKeyPoint modelKeyPoints, Mat observedImageMat, VectorOfKeyPoint observedKeyPoint, VectorOfVectorOfDMatch matches, Mat mask, Mat homography)
        {
            Mat result = new Mat();
            Features2DToolbox.DrawMatches(modelImageMat, modelKeyPoints, observedImageMat, observedKeyPoint, matches, result, new MCvScalar(255, 0, 0), new MCvScalar(255, 255, 255), mask);

            //draw a rectangle along the projected model
            if (homography != null)
            {
                Rectangle rect = new Rectangle(Point.Empty, modelImageMat.Size);
                PointF[] pts = new PointF[]
                {
                    new PointF(rect.Left, rect.Bottom),
                    new PointF(rect.Right, rect.Bottom),
                    new PointF(rect.Right, rect.Top),
                    new PointF(rect.Left, rect.Top)
                };
                pts = CvInvoke.PerspectiveTransform(pts, homography);

                Point[] points = new Point[pts.Length];
                for (int i = 0; i < points.Length; i++)
                    points[i] = Point.Round(pts[i]);

                using (VectorOfPoint vp = new VectorOfPoint(points))
                {
                    CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                }
            }

            return result;
        }

        /// <summary>
        /// Draw the given rectangle on the given image.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="rectangle"></param>
        public void DrawRectangle(Mat image, Rectangle rectangle)
        {
            //using (UMat imageUmat = image.GetUMat(AccessType.ReadWrite))
            {
                CvInvoke.Rectangle(image, rectangle, new MCvScalar(255, 0, 0, 255));
            }
        }

        /// <summary>
        /// Draw the given keypoints on the given image.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="keypoints"></param>
        /// <returns></returns>
        public Mat DrawKeypoints(Mat image, MKeyPoint[] keypoints)
        {
            // Clone the image so we can work on an intermediate
            var result = image.Clone();
            //using (UMat imageUmat = image.GetUMat(AccessType.Read))
            //using (UMat resultUmat = result.GetUMat(AccessType.ReadWrite))
            using (VectorOfKeyPoint keypointsVector = new VectorOfKeyPoint(keypoints))
            {
                Features2DToolbox.DrawKeypoints(image, keypointsVector, result, new Bgr(255, 0, 0));
            }

            return result;
        }
    }
}
