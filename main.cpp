#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
//#include <unistd.h>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include <list>

using namespace std;
using namespace cv;

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}

class GaborFilterProject
{	
public:

	GaborFilterProject(string fName, Mat imgGray, Mat imgColor, int K,
			bool bVerbose, bool bDumpConv, bool bDumpKernel, bool bDumpKMean, bool bDumpHistogram) : 
				mK(K), mbVerbose(bVerbose), mbDumpConv(bDumpConv), mbDumpKernel(bDumpKernel), mbDumpHistogram(bDumpHistogram)
	{
		srand(time(NULL));
		string dName = to_string(rand());
	
		mFolderName = string("Folder_") + dName;
		system((string("mkdir ") + mFolderName).c_str());

		cout << mFolderName + string("/name.txt") << endl;
		std::ofstream outputFile(mFolderName + string("/name.txt"));
		outputFile << fName << endl;
		outputFile.close();

		// Save original image in grayscale and color
		mSrcGrayMat = imgGray;
		mSrcColorMat = imgColor;
	}

	void InvokeKMeanCluster()
	{
		Mat KMeanImage = imread("KMean.png", 0);
		int histSize = 256;
		float range[] = {0, 256} ;
		const float* histRange = { range };
	    bool uniform = true; bool accumulate = false;

		Mat outMat;
		calcHist(&KMeanImage, 1, 0, Mat(), outMat, 1, &histSize, &histRange, uniform, accumulate);

	    // Draw the histograms for B, G and R
	    int hist_w = 512; int hist_h = 400;
	    int bin_w = cvRound( (double) hist_w/histSize );
		
		Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	    normalize(outMat, outMat, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		
		/// Draw for each channel
		for(int i = 1; i < histSize; i++)
		{
			line(histImage, Point( bin_w*(i-1), hist_h - cvRound(outMat.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(outMat.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
        }

	  /// Display
	  //namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	  //imshow("calcHist Demo", histImage );

           //  waitKey(0);

	return;

		int const_size = 100;

		cv::Size sizees = KMeanImage.size();

		int num_h = ceil(sizees.height / const_size),
			num_w = ceil(sizees.width / const_size);

		Mat NewMat(const_size*const_size, num_h * num_w, KMeanImage.type());

		Mat Part(1, const_size*const_size, KMeanImage.type());
		int i = 0;
				
		for (int h = 0; h < sizees.height; h += const_size)
		{
			Mat CurrRow = KMeanImage.rowRange(h, min(h+const_size, sizees.height));
			
			for (int w = 0; w < sizees.width; w += const_size)
			{
				if (w + const_size > sizees.width) continue;
				Mat CurrCol = CurrRow.colRange(w, min(w+const_size, sizees.width));
					
				CurrCol.copyTo(Part);

				Part = Part.reshape(1, CurrCol.size().area());
				if (Part.size().area() < const_size * const_size) continue;

				Part.copyTo(NewMat.rowRange(0, Part.size().area()).colRange(i, i+1));
				imshow("a,", CurrCol);
				waitKey(0);
				++i;
			}
		}
		
		cout << __LINE__ << endl;
		int attempts2 = 3;
		double eps2 = 0.01;
		NewMat.convertTo(NewMat, CV_32F);

		// ofstream fstr = ofstream::open("hist.txt");
		// fstr << 

//		int sizes[] = {imgGray.size().height, imgGray.size().width};
		

		Mat KMeanOut2(num_h, num_w, CV_32F);

		kmeans(NewMat.t(), 20, KMeanOut2, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, attempts2, eps2), attempts2, KMEANS_PP_CENTERS);


		cout << KMeanOut2.size() << " or " << num_h << " and " << num_w << endl;

		int sizees2[] = {num_h, num_w};
		KMeanOut2 = KMeanOut2.reshape(1, 2, sizees2);

		double maxx;
		cv::minMaxLoc(KMeanOut2, 0, &maxx);
		cout << "WOW: " << maxx << endl;

		KMeanOut2 *= 255.0 / maxx;

		// Display
		//namedWindow("calcHist", CV_WINDOW_AUTOSIZE);
		//imshow("calcHist", KMeanOut2);

		//waitKey(0);

		imwrite("tryres40.png", KMeanOut2);

		return;
	}
		
	void InvokeFullScanner()
	{
		/*
		bool mbVerbose;
		bool mbDumpConv; 
		bool mbDumpKernel;
		bool mbDumpKMean;
		bool mbDumpHistogram;
		*/

		// Use real size
		cv::Size theSize = mSrcGrayMat.size();

		// Dest Z value
		mDestZMat = Mat(theSize.width, theSize.height, CV_32F);

		// Sqrt(2)
		const double sqrt2 = sqrt(2);

		int MaxSigma = 5;
		int MaxKernel = int (pow(sqrt2, MaxSigma + 4)); 
		
		// Mat that saves all permutations
		//theSize.height -= MaxKernel;
		//theSize.width -= MaxKernel;

		int theArea = theSize.area();

		Mat AllMats(theArea, (MaxSigma + 1) * 18, CV_32F, cv::Scalar(0));

		// For SigmaI, i = 1 : 12
		for (mSigmaFactor = 0; mSigmaFactor <= MaxSigma; ++mSigmaFactor)
		{
			// Sigma = 4*2^(i/2)
			mSigma = pow(sqrt2, mSigmaFactor + 4); 
			
			// Half Kernel size
			int KernelSize = 4 * ceil(mSigma) + 1;
			

			// For ThetaI, i = 0 : 17
			for (mThetaFactor = 0; mThetaFactor < 18; ++mThetaFactor)
			{
				// Theta = i * pi / 18
				mTheta = mThetaFactor * CV_PI / 18;
								
				// Create kernel using members and permute
				Mat Kernel = makeKernel();
				// Use resize instead...

				int diff =  floor(Kernel.size().width / 2);

				// Move data to temporary mat
				mSrcGrayMat.copyTo(mDestMat);
				// Replicate borders to increase accuracy
				copyMakeBorder(mDestMat, mDestMat, diff, diff, diff, diff, BORDER_REPLICATE);
				
				// Apply filter
				filter2D(mDestMat, mDestMat, CV_32F, Kernel);
				
				// Remove new borders and copy to ZMat
				mDestMat.rowRange(diff, theSize.height+diff).colRange(diff, theSize.width+diff).copyTo(mDestZMat);
				
				// Z value computation
				Mat meanMat, stdMat;
				cv::meanStdDev(mDestZMat, meanMat, stdMat); 
				
				mDestZMat = (mDestZMat - meanMat) / stdMat;
				
				// Copy to whole matrix
				int col = mSigmaFactor * 18 + mThetaFactor;
				
		
				mDestZMat.reshape(1, theArea).copyTo(AllMats.rowRange(0, theArea).colRange(col, col + 1));
				cout << "Done " << mThetaFactor << " and " << mSigmaFactor << endl;
				
				if (true && (true || mbDumpKernel || mbDumpConv))
				{
					string strEnd = string("_") + to_string(mThetaFactor) + string("_") + to_string(mSigmaFactor) + ".png";
				
					// Gernerate kernel to display
					cv::Size kSize(theSize.width / 20, theSize.height / 20);
					kSize.height;
					kSize.width;
	
					double mmin, mmax;
					cv::minMaxIdx(Kernel, &mmin, &mmax);				

					Kernel -= mmin;
					Kernel *= 255.0 / (mmax - mmin);
					resize(Kernel, Kernel, kSize);

					if (true || mbDumpKernel)
					{
						imwrite(mFolderName + "/Kernel" + strEnd, Kernel);
					}

					if (true || mbDumpConv)
					{
						cv::minMaxIdx(mDestZMat, &mmin, &mmax);
						mDestZMat -= mmin;
						mDestZMat *= (255.0 / (mmax - mmin));
						Kernel.copyTo(mDestZMat.colRange(1, kSize.width + 1).rowRange(1, kSize.height + 1));
						imwrite(mFolderName + "/Kernel_applied" + strEnd, mDestZMat);

						
						// Combine with color image
						cvtColor(mDestZMat, mDestZMat, CV_GRAY2BGR);
						mDestZMat.convertTo(mDestZMat, mSrcColorMat.type());

						addWeighted(mDestZMat, 0.5, mSrcColorMat, 0.5, 0.0, mDestZMat);
						imwrite(mFolderName + "/Kernel_applied_col" + strEnd, mDestZMat);
					}
				}
				
			}
		}

		Wait();
		
		if (true || mbDumpKMean)
		{
			int attempts = 1;
			double eps = 0.01;
			cout << "D\n";
			int sizes[] = {mSrcGrayMat.size().height, mSrcGrayMat.size().width};

			Wait();

			mSrcGrayMat.release();
			mSrcColorMat.release();
			mDestZMat.release();
			mDestMat.release();

			Wait();

			Mat KMeanOut;
			kmeans(AllMats, mK, KMeanOut, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, attempts, eps), attempts, KMEANS_PP_CENTERS);
			cout << "F\n";	
			//AllMats.release();
			Wait();

			//AllMats.release();

			Wait();
			double maxp, maxv;
			cv::minMaxLoc(KMeanOut, &maxp, &maxv);
			KMeanOut = KMeanOut.reshape(1, 2, sizes) * 12;
			
			Wait();
			cout << "W\n";
			imwrite(mFolderName + "/KMean.png", KMeanOut);
			Wait();
		//	InvokeKMeanCluster();
		}
		Wait();
	}

protected:

	void Wait()
	{
		static int i = 1;
		cout << "_____ " << (i++) << " _____" << endl;
		//sleep(2);
	}

	static void doProcess(int, void* _this)
	{
		((GaborFilterProject*)_this)->Process();
	}

	void Process()
	{
	}


	Mat makeKernel()
	{
		double xTheta, yTheta;
		int KernelSize = 4 * ceil(mSigma) + 1;
		int KernelCenter = 2 * ceil(mSigma);

		// For cosine version
		// Gamma set to 1
		double Lambda = 2 * mSigma; // need mSigma for sine version
		double Psi = 0;				// Phase is 0

		Mat Kernel(KernelSize, KernelSize, CV_32F);

		for (int y = -KernelCenter; y <= KernelCenter; ++y)
		{
			for (int x = -KernelCenter; x <= KernelCenter; ++x)
			{
				xTheta = x * cos(mTheta) + y * sin(mTheta);
				yTheta = -x * sin(mTheta) + y * cos(mTheta);

				Kernel.rowRange(KernelCenter + y, KernelCenter + y + 1)
					.colRange(KernelCenter + x, KernelCenter + x + 1) = 
						exp(-0.5 * (pow(xTheta, 2) + pow(yTheta, 2)) /
						pow(mSigma, 2)) * sin(2 * CV_PI * xTheta / Lambda + Psi);
			}
		}

		return Kernel;
	}

private:	

	// Number of clusters
	int mK;

	// Folder name
	string mFolderName;

	// The size of window we would like to present in Win32 (not kernel window!)
	int mWinWidth;
	int mWinHeight;

	// Sigma factor: i = 1 : 12 (so sigma = 4 * 2 ^ (i/2))
	int mSigmaFactor;
	
	// Theta factor: i = 0 : 17 (so theta = i * pi / 18)
	int	mThetaFactor;

	// Sigma value
	double mSigma;

	// Theta value
	double mTheta;

	// Original and convoluted Mat
	Mat mSrcGrayMat;
	Mat mSrcColorMat;
	Mat mDestMat;
	Mat mDestZMat;

	// Flags
	bool mbVerbose;
	bool mbDumpConv; 
	bool mbDumpKernel;
	bool mbDumpKMean;
	bool mbDumpHistogram;
};

class ProcessResults
{
public:
	ProcessResults(string Folder) : mFolder(Folder + "/") { }

	void Invoke()
	{
		cout << "Folder: " << mFolder << endl;
		for (int SigmaFactor = 0; SigmaFactor <= 12; ++SigmaFactor)
		{
			cout << "Sigma: " << SigmaFactor << endl;
			Mat compute(18000, 24000, CV_32F, 0);	
			for (int ThetaFactor = 0; ThetaFactor < 18; ++ThetaFactor)
			{
				string FName = mFolder + to_string(SigmaFactor) + \
					string("_") + to_string(ThetaFactor) + ".png";
				cout << FName << endl;
				Mat image = imread(FName);
				cout << __LINE__ << endl;
				cvtColor(image, image, CV_BGR2GRAY);
				cout << __LINE__ << endl;
                		image.convertTo(image, CV_32F, 1.0 / 255, 0);
				cout << __LINE__ << endl;
				compute += image;
				cout << __LINE__ << endl;
			}

			compute *= (1.0f / 18);
			string FName = mFolder + to_string(SigmaFactor) + string("_sigmas.png");
			imwrite(FName, compute);
		}

		for (int ThetaFactor = 0; ThetaFactor < 18; ++ThetaFactor)
		{
			cout << "Theta: " << ThetaFactor << endl;
			Mat compute(18000, 24000, CV_32F, 0);
			for (int SigmaFactor = 0; SigmaFactor <= 12; ++SigmaFactor)
			{
				string FName = mFolder + to_string(SigmaFactor) + \
					string("_") + to_string(ThetaFactor) + ".png";
				Mat image = imread(FName);
				cvtColor(image, image, CV_BGR2GRAY);
				image.convertTo(image, CV_32F, 1.0 / 255, 0);
				compute += image;
		  	}	
			
			compute *= (1.0f / 13);
			string FName = mFolder + to_string(ThetaFactor) + string("_thetas.png");
			imwrite(FName, compute);
		}		
	}

private:
	string mFolder;
};

int main(int argc, char *argv[])
{
//	int argc = 3;
//	char *argv[] = {"", "-file", "Slice.tif"};

        cout << "App -K <k> -dump_conv -dump_kernel -dump_histogram -path <dest> -file <img>" << endl;         

	bool bVerbose = true;
	bool bDumpConv = false;
	bool bDumpKernel = false;
	bool bDumpKMean = false;
	bool bDumpHistogram = false;
	int K = 10;
	string strDestPath = ".";
	list<string> listFiles;

	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-K"))
		{
			++i;
			assert(i <= argc);
			K = atoi(argv[i]);
		}

		else if (!strcmp(argv[i], "-dump_conv"))
			bDumpConv = true;

		else if (!strcmp(argv[i], "-dump_kernel"))
			bDumpKernel = true;

		else if (!strcmp(argv[i], "-dump_histogram"))
			bDumpHistogram = true;

		else if (!strcmp(argv[i], "-path"))
		{
			++i;
			assert(i <= argc);
			strDestPath = argv[i];
		}

		else if (!strcmp(argv[i], "-file"))
		{
			++i;
			assert(i <= argc);
			listFiles.push_front(argv[i]);
		}

		else 
			cout << "Invalid argument. Exiting." << endl;
	}

	for (list<string>::iterator it = listFiles.begin();
		it != listFiles.end();
		++it)
	{
		string fName = *it;
		cout << "Processing image: " << fName << endl;
		
		Mat imgColor = imread(fName, 1);
		imgColor.convertTo(imgColor, CV_32F);
		
		Mat imgGray = imread(fName, 0);
		imgGray.convertTo(imgGray, CV_32F);
		
		if (imgGray.empty() || imgColor.empty())
		{
			cout << "Invalid image or format! Skipping..." << endl;
			continue;
		}

		GaborFilterProject gabor_project(fName, imgGray, imgColor, K,
			bVerbose, bDumpConv, bDumpKernel, bDumpKMean, bDumpHistogram);

		gabor_project.InvokeFullScanner();
	}

    return 0;
}
