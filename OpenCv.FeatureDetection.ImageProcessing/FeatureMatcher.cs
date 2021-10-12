using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace OpenCv.FeatureDetection.ImageProcessing
{
    public class FeatureMatcher
    {
        static FeatureMatcher()
        {
            CvInvoke.UseOpenCL = true;
        }

        public FeatureMatcher()
        {

        }

        /// <summary>
        /// Apply Brightness and Contrast adjustments to the given image.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="brightness"></param>
        /// <param name="contrast"></param>
        /// <returns></returns>
        public Mat ApplyBrightnessContrast(Mat image, byte brightness, byte contrast)
        {
            var intermediateImage = image.Clone();
            using (UMat intermediateUmat = intermediateImage.GetUMat(AccessType.ReadWrite))
            {
                if (brightness != 0)
                {
                    int shadow, highlight;
                    if (brightness > 0)
                    {
                        shadow = brightness;
                        highlight = 255;
                    }
                    else
                    {
                        shadow = 0;
                        highlight = brightness + 255;
                    }

                    double alpha = (highlight - shadow) / 255d;
                    double gamma = shadow;

                    CvInvoke.AddWeighted(image, alpha, image, 0, gamma, intermediateUmat);
                }

                if (contrast != 0)
                {
                    double f = 131f * (contrast + 127f) / (127f * (131f - contrast));
                    double alpha = f;
                    double gamma = 127f * (1f - f);

                    CvInvoke.AddWeighted(intermediateUmat, alpha, intermediateUmat, 0, gamma, intermediateUmat);
                }

                var result = new Mat();
                intermediateUmat.CopyTo(result);

                return result;
            }
        }

        /// <summary>
        /// Detect features in the given image using the given feature detector.
        /// </summary>
        /// <param name="featureDetector"></param>
        /// <param name="image"></param>
        /// <returns></returns>
        public IEnumerable<MKeyPoint> DetectFeatures(Feature2D featureDetector, Mat image)
        {
            using (UMat imageUmat = image.GetUMat(AccessType.Read))
            {
                var observedImageKeypointArray = featureDetector.Detect(imageUmat);
                return observedImageKeypointArray;
            }
        }

        /// <summary>
        /// Match the given images using the given detector, extractor, and matcher, calculating and returning homography.
        /// 
        /// The given detector is used for detecting keypoints.
        /// The given extractor is used for extracting descriptors.
        /// The given matcher is used for computing matches.
        /// 
        /// Detection and matching will be done in two separate stages.
        /// 
        /// The Mat and Vector... properties of this result are unmanaged - it is assumed the caller will dispose results.
        /// </summary>
        /// <param name="featureDetector"></param>
        /// <param name="featureExtractor"></param>
        /// <param name="matcher"></param>
        /// <param name="observedImage"></param>
        /// <param name="modelImage"></param>
        /// <returns></returns>
        public MatchFeaturesResult MatchFeatures(Feature2D featureDetector, Feature2D featureExtractor, DescriptorMatcher matcher, Mat observedImage, Mat modelImage)
        {
            using (UMat observedImageUmat = observedImage.GetUMat(AccessType.Read))
            using (UMat modelImageUmat = modelImage.GetUMat(AccessType.Read))
            {
                // Detect keypoints
                var observedImageKeypoints = featureDetector.Detect(observedImageUmat);
                var modelImageKeypoints = featureDetector.Detect(modelImageUmat);

                var observedDescriptors = new Mat();
                var modelDescriptors = new Mat();

                var observedKeypointVector = new VectorOfKeyPoint(observedImageKeypoints);
                var modelKeypointVector = new VectorOfKeyPoint(modelImageKeypoints);

                // Compute descriptors
                featureExtractor.Compute(observedImageUmat, observedKeypointVector, observedDescriptors);
                featureExtractor.Compute(modelImageUmat, modelKeypointVector, modelDescriptors);

                // Match descriptors
                matcher.Add(modelDescriptors);

                var matches = new VectorOfVectorOfDMatch();
                matcher.KnnMatch(observedDescriptors, matches, 2);

                // Filter matches based on ratio
                //matches = LowesFilter(matches);

                var mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                mask.SetTo(new MCvScalar(255));

                Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);
                Mat homography = null;
                var nonZeroCount = CvInvoke.CountNonZero(mask);
                if (nonZeroCount >= 4)
                {
                    nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeypointVector, observedKeypointVector, matches, mask, 1.5, 20);
                    //nonZeroCount = FilterByDistanceFromAveragePoint(matches, mask, nonZeroCount, observedKeypointVector, modelImage);
                    if (nonZeroCount >= 4)
                    {
                        // Attempt to remove outliers
                        var modelPoints = new PointF[matches.Size];
                        var observedPoints = new PointF[matches.Size];

                        for (var index = 0; index < matches.Size; index++)
                        {
                            var match = matches[index];
                            var modelPoint = modelKeypointVector[match[0].TrainIdx].Point;
                            modelPoints[index] = modelPoint;

                            var observedPoint = observedKeypointVector[match[0].QueryIdx].Point;
                            observedPoints[index] = observedPoint;
                        }

                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeypointVector, observedKeypointVector, matches, mask, 2);
                        //homography = CvInvoke.EstimateAffine2D(modelPoints, observedPoints, inliners: mask);
                    }
                }

                var result = new MatchFeaturesResult(observedKeypointVector, observedDescriptors, modelKeypointVector, modelDescriptors, matches, mask, homography);
                return result;
            }
        }

        /// <summary>
        /// Match the given images using the given detector/extractor and matcher, calculating and returning homography.
        /// 
        /// The given detector is used for detecting keypoints and extraction of descriptors.
        /// The given matcher is used for computing matches.
        /// 
        /// Detection and matching will be done in one stage. This is ideal for combination detector/extractors like ORB.
        /// 
        /// The Mat and Vector... properties of this result are unmanaged - it is assumed the caller will dispose results.
        /// </summary>
        /// <param name="featureDetector"></param>
        /// <param name="matcher"></param>
        /// <param name="observedImage"></param>
        /// <param name="modelImage"></param>
        /// <returns></returns>
        public MatchFeaturesResult MatchFeatures(Feature2D featureDetector, DescriptorMatcher matcher, Mat observedImage, Mat modelImage)
        {
            using (UMat observedImageUmat = observedImage.GetUMat(AccessType.Read))
            using (UMat modelImageUmat = modelImage.GetUMat(AccessType.Read))
            {
                var observedKeypointVector = new VectorOfKeyPoint();
                var modelKeypointVector = new VectorOfKeyPoint();

                var observedDescriptors = new Mat();
                var modelDescriptors = new Mat();

                // Detect keypoints
                featureDetector.DetectAndCompute(observedImageUmat, null, observedKeypointVector, observedDescriptors, false);
                featureDetector.DetectAndCompute(modelImageUmat, null, modelKeypointVector, modelDescriptors, false);

                // Match descriptors
                matcher.Add(modelDescriptors);

                var matches = new VectorOfVectorOfDMatch();
                matcher.KnnMatch(observedDescriptors, matches, 2);

                // Filter matches based on ratio
                //matches = LowesFilter(matches);

                // As implied by use of CountNonZero, zero'd results are those excluded and/or invalid.
                var mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                mask.SetTo(new MCvScalar(255));

                Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);
                Mat homography = null;
                var nonZeroCount = CvInvoke.CountNonZero(mask);
                if (nonZeroCount >= 4)
                {
                    nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeypointVector, observedKeypointVector, matches, mask, 1.5, 20);
                    // This isn't truly filtering outliers - if the "weighted center" of the matches is sufficiently far off, then it's just hosed.
                    // I think this indicates an excess of weak features resulting in poor matching.
                    //nonZeroCount = FilterByDistanceFromAveragePoint(matches, mask, nonZeroCount, observedKeypointVector, modelImage);

                    if (nonZeroCount >= 4)
                    {

                        //var modelPoints = new PointF[matches.Size];
                        //var observedPoints = new PointF[matches.Size];

                        //for (var index = 0; index < matches.Size; index++)
                        //{
                        //    var match = matches[index];
                        //    var modelPoint = modelKeypointVector[match[0].TrainIdx].Point;
                        //    modelPoints[index] = modelPoint;

                        //    var observedPoint = observedKeypointVector[match[0].QueryIdx].Point;
                        //    observedPoints[index] = observedPoint;
                        //}

                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeypointVector, observedKeypointVector, matches, mask, 2);
                        //homography = CvInvoke.EstimateAffine2D(observedPoints, modelPoints, inliners: mask);
                    }
                }

                var result = new MatchFeaturesResult(observedKeypointVector, observedDescriptors, modelKeypointVector, modelDescriptors, matches, mask, homography);
                return result;
            }
        }

        /// <summary>
        /// Filter the given matches by calculating the average point of valid matches (clustered center), setting boundaries from that point, and excluding any result outside the boundary.
        /// The boundary is calculated as (center - model width) <= x <= (center + model width); (center - model height) <= y <= (center + model height).
        /// </summary>
        /// <param name="matches"></param>
        /// <param name="mask">Mask containing values indicating valid matches. 0 for invalid, 255 for valid. This mask is updated as part of this process.</param>
        /// <param name="validMatchCount">Valid match count before applying this filter.</param>
        /// <param name="observedKeypointVector"></param>
        /// <param name="modelImage"></param>
        /// <returns>Number of remaining accepted matches in our mask.</returns>
        private int FilterByDistanceFromAveragePoint(VectorOfVectorOfDMatch matches, Mat mask, int validMatchCount, VectorOfKeyPoint observedKeypointVector, Mat modelImage)
        {
            // Attempt to derive an average center of detected keypoints
            double x = 0, y = 0;

            // EmguCV does not provide for a clean access layer for underlying data - it must be copied to a CLR array and back, or accessed as unmanaged (unsafe) memory.
            // Mat.CopyTo() was discovered through EmguCV documentation: https://www.emgu.com/wiki/files/4.5.3/document/html/2ec33afb-1d2b-cac1-ea60-0b4775e4574c.htm
            var maskData = new byte[mask.Rows];
            mask.CopyTo(maskData);

            for (var index = 0; index < matches.Size; index++)
            {
                // If this was a rejected match, skip it
                if (maskData[index] == 0) continue;

                var sourceKeypointIndex = matches[index][0].QueryIdx;
                var sourceImagePoint = observedKeypointVector[sourceKeypointIndex];
                x += sourceImagePoint.Point.X;
                y += sourceImagePoint.Point.Y;
            }

            x /= validMatchCount;
            y /= validMatchCount;

            // anything outside 2x model size from average center should be excluded
            validMatchCount = 0;
            for (var index = 0; index < matches.Size; index++)
            {
                if (maskData[index] == 0) continue;

                var sourceKeypointIndex = matches[index][0].QueryIdx;
                var sourceImagePoint = observedKeypointVector[sourceKeypointIndex];

                var left = x - modelImage.Width;
                var right = x + modelImage.Width;
                var top = y + modelImage.Height;
                var bottom = y - modelImage.Height;

                var isInBounds = left <= sourceImagePoint.Point.X && sourceImagePoint.Point.X <= right &&
                    bottom <= sourceImagePoint.Point.Y && sourceImagePoint.Point.Y <= top;
                if (!isInBounds)
                {
                    // Unset this as a match in our mask
                    maskData[index] = 0;
                }
                validMatchCount++;
            }

            // Copy the mask data back to the mask
            mask.SetTo(maskData);

            return validMatchCount;
        }

        private VectorOfVectorOfDMatch LowesFilter(VectorOfVectorOfDMatch matches)
        {
            var results = new VectorOfVectorOfDMatch();
            for (var index = 0; index < matches.Size; index++)
            {
                if (matches[index][0].Distance < 0.75f * matches[index][1].Distance)
                {
                    results.Push(matches[index]);
                }
            }
            return results;
        }
    }
}
