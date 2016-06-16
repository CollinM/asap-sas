package collinm.asap;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.javatuples.Triplet;

import collinm.asap.util.LoadAsapData;
import collinm.asap.util.Output;
import collinm.util.ConfusionMatrix;

public class BaselineTfIdf {

	public static void main(String[] args) {
		// Parse CLI args
		Path inputFile = Paths.get(args[0]);
		Path outDir = Paths.get(args[1]);
		int trees = Integer.parseInt(args[2]);
		int depth = Integer.parseInt(args[3]);
		
		// Setup Spark
		SparkConf conf = new SparkConf().setAppName("Baseline_TF-IDF");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SQLContext sql = new SQLContext(jsc);
		
		// Read data
		System.out.println("Reading data");
		DataFrame all = LoadAsapData.processedData(inputFile, sql, false);
		DataFrame[] split = all.randomSplit(new double[] {0.8, 0.2}, 42);
		DataFrame train = split[0];
		DataFrame test = split[1];
		
		// Build pipeline
		StringIndexer targetIndexer = new StringIndexer()
				.setInputCol("Score1")
				.setOutputCol("Score1_index");
		StringIndexerModel targetIndexerModel = targetIndexer.fit(train);
		IndexToString targetUnIndexer = new IndexToString()
				.setInputCol("pred-score-index")
				.setOutputCol("pred-score")
				.setLabels(targetIndexerModel.labels());
		
		Tokenizer tokenizer = new Tokenizer().setInputCol("EssayText").setOutputCol("tokens");
		HashingTF hashingTF = new HashingTF()
				  .setInputCol("tokens")
				  .setOutputCol("tf")
				  .setNumFeatures(4000);
		IDF idf = new IDF().setInputCol("tf").setOutputCol("features");
		RandomForestClassifier rfc = new RandomForestClassifier()
				.setNumTrees(trees)
				.setMaxDepth(depth)
				.setFeatureSubsetStrategy("auto")
				.setLabelCol("Score1_index")
				.setPredictionCol("pred-score-index");
		
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { targetIndexer, tokenizer, hashingTF, idf, rfc, targetUnIndexer });
		
		// Train model
		System.out.println("Training model");
		train.cache();
		PipelineModel model = pipeline.fit(train);
		train.unpersist();

		// Evaluate test data
		System.out.println("Testing model");
		test.persist();
		DataFrame predxns = model.transform(test);
		test.unpersist();

		// Write performance data
		System.out.println("Writing performance data");
		ConfusionMatrix metrics = new ConfusionMatrix(LoadAsapData.SCORES);
		for (Row r : predxns.select("Score1", "pred-score").collect())
			metrics.increment(r.getString(0), r.getString(1));
		metrics.writeMatrixCSV(outDir);
		metrics.writeMetricsCSV(outDir);

		List<Triplet<String, String, String>> perfData = new ArrayList<>();
		for (Row r : predxns.select("ID", "Score1", "pred-score").collect())
			perfData.add(Triplet.with(r.getString(0), r.getString(1), r.getString(2)));
		Output.writePerformanceData(outDir, "performance.csv", perfData);
	}
}
