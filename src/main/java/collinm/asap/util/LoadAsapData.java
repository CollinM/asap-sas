package collinm.asap.util;

import java.nio.file.Path;

import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class LoadAsapData {

	public final static String[] SCORES = new String[] { "0", "1", "2", "3" };
	
	private static final StructField ID = new StructField("ID", DataTypes.StringType, false, Metadata.empty());
	private static final StructField ESSAY_SET = new StructField("EssaySet", DataTypes.StringType, false, Metadata.empty());
	private static final StructField SCORE1_D = new StructField("Score1", DataTypes.DoubleType, false, Metadata.empty());
	private static final StructField SCORE1_S = new StructField("Score1", DataTypes.StringType, false, Metadata.empty());
	private static final StructField SCORE2_D = new StructField("Score2", DataTypes.DoubleType, false, Metadata.empty());
	private static final StructField SCORE2_S = new StructField("Score2", DataTypes.StringType, false, Metadata.empty());
	private static final StructField ESSAY_TEXT = new StructField("EssayText", DataTypes.StringType, false, Metadata.empty());

	public static DataFrame baseData(Path filePath, SQLContext sql, boolean doubleScores) {
		StructType schema = doubleScores
				? new StructType(new StructField[] { ID, ESSAY_SET, SCORE1_D, SCORE2_D, ESSAY_TEXT })
				: new StructType(new StructField[] { ID, ESSAY_SET, SCORE1_S, SCORE2_S, ESSAY_TEXT });
		
		DataFrame df = sql.read()
				.format("com.databricks.spark.csv")
				.option("delimiter", "\t")
				.schema(schema)
				.load(filePath.toString());
		return df;
	}
	
	public static DataFrame processedData(Path filePath, SQLContext sql, boolean doubleScores) {
		StructType schema = doubleScores
				? new StructType(new StructField[] { ID, SCORE1_D, SCORE2_D, ESSAY_TEXT })
				: new StructType(new StructField[] { ID, SCORE1_S, SCORE2_S, ESSAY_TEXT });
		
		DataFrame df = sql.read()
				.format("com.databricks.spark.csv")
				.option("delimiter", "\t")
				.schema(schema)
				.load(filePath.toString());
		return df;
	}
}
