import org.apache.spark.mllib.clustering.dbscan.DBSCAN
import org.apache.spark.mllib.linalg.Vectors
import com.amazonaws.services.s3._

object DBSCANSample {

  def main(args: Array[String]) {

//    val conf = new SparkConf().setAppName("RDK DBSCAN")
//    val sc = new SparkContext(conf)

    val data = sc.textFile("fit_rdd.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.replaceAll("[\\(\\[\\]\\)]","")).map(_.toDouble))).cache()
    val model = DBSCAN.fit(
      eps = 0.01f,
      minPoints = 5,
      data = parsedData)

    model.saveAsTextFile("dbscan.txt")

//    sc.stop()
  }
}