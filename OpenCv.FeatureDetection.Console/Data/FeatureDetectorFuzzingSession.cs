using System;
using System.Collections.Generic;

namespace OpenCv.FeatureDetection.Console.Data
{
    public class FeatureDetectorFuzzingSession
    {
        public int Id { get; set; }
        public DateTime StartDateTime { get; set; }
        public DateTime? EndDateTime { get; set; }
        
        public IEnumerable<FeatureDetectionResult> FeatureDetectionResults { get; set; }
    }
}
