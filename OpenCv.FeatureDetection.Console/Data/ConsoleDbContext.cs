using Microsoft.EntityFrameworkCore;

namespace OpenCv.FeatureDetection.Console.Data
{
    public class ConsoleDbContext : DbContext
    {
        public DbSet<FeatureDetectorFuzzingSession> FeatureDetectorFuzzingSessions { get; set; }
        public DbSet<FeatureDetectionResult> FeatureDetectionResults { get; set; }

        private readonly string _databaseFilePath;

        public ConsoleDbContext(string databaseFilePath)
        {
            _databaseFilePath = databaseFilePath;
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlite($"Data Source={_databaseFilePath};");

            base.OnConfiguring(optionsBuilder);
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            var featureDetectorFuzzingSessionEntity = modelBuilder.Entity<FeatureDetectorFuzzingSession>();
            featureDetectorFuzzingSessionEntity.ToTable("FeatureDetectionFuzzingSession");

            featureDetectorFuzzingSessionEntity.HasKey(x => x.Id);

            featureDetectorFuzzingSessionEntity.HasMany(x => x.FeatureDetectionResults).WithOne(x => x.FuzzingSession).HasForeignKey(x => x.FeatureDetectorFuzzingSessionId);

            var featureDetectionResultsEntity = modelBuilder.Entity<FeatureDetectionResult>();
            featureDetectionResultsEntity.ToTable("FeatureDetectionResult");
            featureDetectionResultsEntity.HasKey(x => x.Id);

            base.OnModelCreating(modelBuilder);
        }
    }
}
