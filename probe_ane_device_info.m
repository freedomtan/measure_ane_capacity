#import <Foundation/Foundation.h>
#import <objc/runtime.h>

@interface _ANEDeviceInfo : NSObject
+ (NSInteger)numANECores;
+ (NSInteger)numANEs;
+ (NSString *)aneArchitectureType;
+ (NSString *)aneSubType;
+ (NSString *)aneSubTypeVariant;
+ (NSString *)aneSubTypeProductVariant;
+ (NSInteger)aneBoardType;
@end

int main() {
  @autoreleasepool {
    printf("numANEs: %ld\n", (long)[_ANEDeviceInfo numANEs]);
    printf("numANECores: %ld\n", (long)[_ANEDeviceInfo numANECores]);
    printf("aneArchitectureType: %s\n", [[_ANEDeviceInfo aneArchitectureType] UTF8String] ?: "null");
    printf("aneSubType: %s\n", [[_ANEDeviceInfo aneSubType] UTF8String] ?: "null");
    printf("aneSubTypeVariant: %s\n", [[_ANEDeviceInfo aneSubTypeVariant] UTF8String] ?: "null");
    printf("aneSubTypeProductVariant: %s\n", [[_ANEDeviceInfo aneSubTypeProductVariant] UTF8String] ?: "null");
    printf("aneBoardType: %ld\n", (long)[_ANEDeviceInfo aneBoardType]);
  }
  return 0;
}
