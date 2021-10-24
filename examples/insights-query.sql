WITH [recent-session-cte] AS (
	SELECT MAX(fdfs.[Id]) AS [MaxId]
	FROM [FeatureDetectionFuzzingSession] fdfs
)
, [recentResultsProjection] AS (
	SELECT fdr.InlierFeatureCount + 3 AS [InlierFeatureMax],
		fdr.InlierFeatureCount - 3 AS [InlierFeatureMin],
		fdr.*
	FROM [FeatureDetectionResult] fdr
	WHERE EXISTS (
		SELECT 1
		FROM [recent-session-cte] rsc
		WHERE rsc.[MaxId] = fdr.[FeatureDetectorFuzzingSessionId]
	)
)
, [hudcte] AS (
	SELECT *
	FROM [recentResultsProjection] rrp
	WHERE rrp.[InputFileName] = 'fishing-hud-bobber-001.jpg'
), [modelcte] AS (
	SELECT *
	FROM [recentResultsProjection] rrp
	WHERE rrp.[InputFileName] = 'model-bobber-001.jpg'
)
, [inlierOverlapCte] AS (
	SELECT *
	FROM [hudcte] hud
	JOIN [modelcte] model
		ON hud.[Algorithm] = model.[Algorithm]
		AND hud.[Iteration] = model.[Iteration]
	WHERE hud.[InlierFeatureCount] >= model.[InlierFeatureMin]
		AND hud.[InlierFeatureCount] <= model.[InlierFeatureMax]
		AND model.[InlierFeatureCount] >= hud.[InlierFeatureMin]
		AND model.[InlierFeatureCount] <= hud.[InlierFeatureMax]
)
SELECT *
FROM [inlierOverlapCte] results
WHERE results.[ExecutionTime] <= 300
AND results.[InlierFeatureCount] <= 30
ORDER BY results.[InlierOutlierRatio] DESC;