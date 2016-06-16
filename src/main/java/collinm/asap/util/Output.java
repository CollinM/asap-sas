package collinm.asap.util;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.javatuples.Triplet;

import com.google.common.base.Joiner;

public class Output {

	public static void writePerformanceData(Path outDir, String fname, List<Triplet<String, String, String>> items) {
		Joiner commaJoiner = Joiner.on(",");
		try {
			Files.createDirectories(outDir);
			BufferedWriter writer = Files.newBufferedWriter(Paths.get(outDir.toString(), fname));
			for (Triplet<String, String, String> tuple : items)
				writer.write(commaJoiner.join(tuple.getValue0(), tuple.getValue1(), tuple.getValue2()) + "\n");
			writer.close();
		} catch (IOException io) {
			io.printStackTrace();
		}
	}
}
