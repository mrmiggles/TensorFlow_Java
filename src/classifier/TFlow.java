package classifier;
import java.util.Arrays;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class TFlow {
	  public static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
		    try (Graph g = new Graph()) {
		      g.importGraphDef(graphDef);
		      try (Session s = new Session(g);
		          Tensor result = s.runner().feed("input", image).fetch("output").run().get(0)) {
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
	  
	  
	 // In the fullness of time, equivalents of the methods of this class should be auto-generated from
	  // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
	  // like Python, C++ and Go.
	  static class GraphBuilder {
	    GraphBuilder(Graph g) {
	      this.g = g;
	    }

	    Output div(Output x, Output y) {
	      return binaryOp("Div", x, y);
	    }

	    Output sub(Output x, Output y) {
	      return binaryOp("Sub", x, y);
	    }

	    Output resizeBilinear(Output images, Output size) {
	      return binaryOp("ResizeBilinear", images, size);
	    }

	    Output expandDims(Output input, Output dim) {
	      return binaryOp("ExpandDims", input, dim);
	    }

	    Output cast(Output value, DataType dtype) {
	      return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
	    }

	    Output decodeJpeg(Output contents, long channels) {
	      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
	          .addInput(contents)
	          .setAttr("channels", channels)
	          .build()
	          .output(0);
	    }

	    Output constant(String name, Object value) {
	      try (Tensor t = Tensor.create(value)) {
	        return g.opBuilder("Const", name)
	            .setAttr("dtype", t.dataType())
	            .setAttr("value", t)
	            .build()
	            .output(0);
	      }
	    }

	    private Output binaryOp(String type, Output in1, Output in2) {
	      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
	    }

	    private Graph g;
	  }	
}
