namespace OpenCv.FeatureDetection.Console.Data
{
    public class FeatureDetectionResult
    {
        public int Id { get; set; }
        
        public string InputFileName { get; set; }
        public string Algorithm { get; set; }
        public int Iteration { get; set; }

        public int InlierFeatureCount { get; set; }
        public int TotalFeatureCount { get; set; }
        public float InlierOutlierRatio { get; set; }

        public long ExecutionTime { get; set; }

        public string Parameters { get; set; }

        public int FeatureDetectorFuzzingSessionId { get; set; }
        public FeatureDetectorFuzzingSession FuzzingSession { get; set; }
    }
}
