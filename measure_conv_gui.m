#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <UIKit/UIKit.h>
#import <time.h>

@interface MPSGraphDevice (ANE)
+ (instancetype)ANEDevice;
@end

// --- Benchmarking Logic (Refactored for GUI) ---
@interface Benchmarker : NSObject
- (void)runAtIndex:(NSInteger)index logHandler:(void (^)(NSString *))log;
@end

@implementation Benchmarker
- (void)runAtIndex:(NSInteger)index logHandler:(void (^)(NSString *))logger {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    logger(@"Error: Failed to get default Metal device.");
    return;
  }

  // Test Configurations
  struct Config {
    bool useANE;
    MPSDataType type;
    NSString *name;
  } configs[] = {
      {false, MPSDataTypeFloat16, @"GPU FP16"},
      {true, MPSDataTypeFloat16, @"ANE FP16"},
      // {false, MPSDataTypeInt8,    @"GPU INT8"}, // Skip GPU INT8
      {true, MPSDataTypeInt8, @"ANE INT8"},
  };

  int configCount = sizeof(configs) / sizeof(configs[0]);
  if (index >= configCount)
    return;

  struct Config cfg = configs[index];
  [self runBench:device
          useANE:cfg.useANE
        dataType:cfg.type
            name:cfg.name
             log:logger];
}

- (void)runBench:(id<MTLDevice>)device
          useANE:(bool)useANE
        dataType:(MPSDataType)dataType
            name:(NSString *)name
             log:(void (^)(NSString *))logger {
  @autoreleasepool {
    MPSGraphDevice *mDev = nil;
    if (useANE) {
      if ([MPSGraphDevice respondsToSelector:@selector(ANEDevice)]) {
        mDev = [MPSGraphDevice ANEDevice];
      } else {
        logger([NSString
            stringWithFormat:@"[%@] Skipped: ANE not supported.", name]);
        return;
      }
    } else {
      mDev = [MPSGraphDevice deviceWithMTLDevice:device];
    }

    MPSGraph *graph = [MPSGraph new];
    NSUInteger B = 1, H = 256, W = 256, Ci = 128, Co = 128, K = 3, L = 20;
    NSArray *inShape = @[ @(B), @(Ci), @(H), @(W) ];
    NSArray *wShape = @[ @(Co), @(Ci), @(K), @(K) ];

    MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                               dataType:dataType
                                                   name:@"in"];
    MPSGraphTensor *cur = input;
    NSUInteger elementSize = (dataType == MPSDataTypeFloat16) ? 2 : 1;

    NSMutableData *wData =
        [NSMutableData dataWithLength:Co * Ci * K * K * elementSize];
    MPSGraphTensor *w = [graph constantWithData:wData
                                          shape:wShape
                                       dataType:dataType];

    for (int i = 0; i < L; i++) {
      MPSGraphConvolution2DOpDescriptor *d = [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:1
                        strideInY:1
                  dilationRateInX:1
                  dilationRateInY:1
                           groups:1
                     paddingStyle:MPSGraphPaddingStyleTF_SAME
                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
      cur = [graph convolution2DWithSourceTensor:cur
                                   weightsTensor:w
                                      descriptor:d
                                            name:nil];
      // Realistic Quantized flow: Int8 -> (Conv) -> Int32/FP16 ->
      // (Select/Scale) -> Int8
      if (dataType == MPSDataTypeInt8) {
        MPSGraphTensor *fp = [graph castTensor:cur
                                        toType:MPSDataTypeFloat16
                                          name:@"dequant"];
        cur = [graph castTensor:fp toType:MPSDataTypeInt8 name:@"requant"];
      }
    }

    NSDictionary *feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:inShape
                                               dataType:dataType]
    };
    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel =
        useANE ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;

    MPSGraphExecutable *exe = [graph compileWithDevice:mDev
                                                 feeds:feeds
                                         targetTensors:@[ cur ]
                                      targetOperations:nil
                                 compilationDescriptor:cd];
    if (!exe) {
      logger([NSString stringWithFormat:@"[%@] Compilation Failed", name]);
      return;
    }

    id<MTLBuffer> iBuf =
        [device newBufferWithLength:B * H * W * Ci * elementSize options:0];
    MPSGraphTensorData *iData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                shape:inShape
                                             dataType:dataType];
    id<MTLCommandQueue> q = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;

    // Warmup
    [exe runWithMTLCommandQueue:q
                    inputsArray:@[ iData ]
                   resultsArray:nil
            executionDescriptor:ed];

    NSUInteger iterations = 20;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
      [exe runWithMTLCommandQueue:q
                      inputsArray:@[ iData ]
                     resultsArray:nil
              executionDescriptor:ed];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double duration =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double avg = duration / iterations;
    double tops = (2.0 * B * H * W * Ci * Co * K * K * L) / (avg * 1e12);

    logger([NSString stringWithFormat:@"[%@] Avg: %.2f ms, Speed: %.4f TOPS",
                                      name, avg * 1000.0, tops]);
  }
}
@end

// --- GUI Implementation ---
@interface ViewController : UIViewController
@property(nonatomic, strong) UITextView *textView;
@property(nonatomic, strong) UIButton *button;
@property(nonatomic, strong) Benchmarker *benchmarker;
@property(nonatomic, assign) BOOL isRunning;
@end

@implementation ViewController
- (void)viewDidLoad {
  [super viewDidLoad];
  self.view.backgroundColor = [UIColor systemBackgroundColor];
  self.benchmarker = [Benchmarker new];
  self.isRunning = NO;

  self.textView =
      [[UITextView alloc] initWithFrame:CGRectInset(self.view.bounds, 20, 100)];
  self.textView.editable = NO;
  self.textView.font = [UIFont monospacedSystemFontOfSize:12
                                                   weight:UIFontWeightRegular];
  self.textView.backgroundColor = [UIColor secondarySystemBackgroundColor];
  self.textView.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  [self.view addSubview:self.textView];

  self.button = [UIButton buttonWithType:UIButtonTypeSystem];
  [self.button setTitle:@"Run Benchmark" forState:UIControlStateNormal];
  [self.button.titleLabel setFont:[UIFont boldSystemFontOfSize:18]];
  self.button.frame = CGRectMake(0, self.view.bounds.size.height - 80,
                                 self.view.bounds.size.width, 50);
  self.button.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleTopMargin;
  [self.button addTarget:self
                  action:@selector(startBenchmark)
        forControlEvents:UIControlEventTouchUpInside];
  [self.view addSubview:self.button];

  [self log:@"Ready to benchmark. Press the button below."];
}

- (void)log:(NSString *)msg {
  if ([NSThread isMainThread]) {
    self.textView.text =
        [self.textView.text stringByAppendingFormat:@"%@\n", msg];
    [self.textView
        scrollRangeToVisible:NSMakeRange(self.textView.text.length, 0)];
  } else {
    dispatch_async(dispatch_get_main_queue(), ^{
      [self log:msg];
    });
  }
}

- (void)startBenchmark {
  if (self.isRunning)
    return;
  self.isRunning = YES;
  self.button.enabled = NO;
  self.textView.text = @"Starting Benchmark...\n";

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    [self.benchmarker runAtIndex:0
                      logHandler:^(NSString *s) {
                        [self log:s];
                      }]; // GPU FP16
    [self.benchmarker runAtIndex:1
                      logHandler:^(NSString *s) {
                        [self log:s];
                      }]; // ANE FP16
    [self.benchmarker runAtIndex:2
                      logHandler:^(NSString *s) {
                        [self log:s];
                      }]; // ANE INT8

    dispatch_async(dispatch_get_main_queue(), ^{
      [self log:@"\nDone!"];
      self.isRunning = NO;
      self.button.enabled = YES;
    });
  });
}
@end

// --- App Entry Point ---
@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property(strong, nonatomic) UIWindow *window;
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
  self.window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
  self.window.rootViewController = [ViewController new];
  [self.window makeKeyAndVisible];
  return YES;
}
@end

int main(int argc, char *argv[]) {
  @autoreleasepool {
    return UIApplicationMain(argc, argv, nil,
                             NSStringFromClass([AppDelegate class]));
  }
}
