package general;

import java.util.Arrays;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Executor {
	 public static float[] executeInceptionGraph(byte[] graphDef, Tensor image, String outputLayerName) {
		    try (Graph g = new Graph()) {
		      g.importGraphDef(graphDef);
		      try (Session s = new Session(g);
		          Tensor result = s.runner().feed("input", image).fetch(outputLayerName).run().get(0)) {
		        final long[] rshape = result.shape();
		        if (result.numDimensions() != 2 || rshape[0] != 1) {
		          throw new RuntimeException(
		              String.format(
		                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
		                  Arrays.toString(rshape)));
		        }
		        int nlabels = (int) rshape[1];
		        return result.copyTo(new float[1][nlabels])[0];
		      }
		    }
		  }
	  
	  
	  public static int maxIndex(float[] probabilities) {
		    int best = 0;
		    for (int i = 1; i < probabilities.length; ++i) {
		      if (probabilities[i] > probabilities[best]) {
		        best = i;
		      }
		    }
		    return best;
		  }	  
}
