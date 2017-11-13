package classifier;

public class TFModels {

	public static String user = "299490";
	public static String getMobilenetDirectory() {
		return "C:\\Users\\" + user + "\\Documents\\TensorFlow\\models\\mobilenet\\";
	}
	
	public static String getMobilenetV1_050_PBName() {
		return "mobilenet_v1_0.50_224\\frozen_graph.pb";
	}
	
	public static String getMobilenetV1_050_Lables(){
		return "mobilenet_v1_0.50_224\\labels.txt";
	}
	
	public static String getMobilenetV1_1_PBName() {
		return "mobilenet_v1_1.0_224\\frozen_graph.pb";
	}
	
	public static String getMobilenetV1_1_Lables(){
		return "mobilenet_v1_1.0_224\\labels.txt";
	}	
	
	public static String getInceptionDirectory() {
		return "C:\\Users\\" + user +  "\\Documents\\TensorFlow\\models\\inception5h\\";
	}
	
	public static String getInceptionPBName() {
		return "tensorflow_inception_graph.pb";
	}
	
	public static String getInceptionLables(){
		return "imagenet_comp_graph_label_strings.txt";
	}	
	
	public static String getInception_v3_Directory() {
		return "C:\\Users\\" + user +  "\\Documents\\TensorFlow\\models\\inception_v3\\2015\\";
	}
	
	public static String getInception_v3_PBName() {
		return "inception_v3.pb";
	}
	
	public static String getInception_v3_Lables(){
		return "imagenet_slim_labels.txt";
	}		
}
