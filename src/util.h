#ifndef UTIL_H
#define UTIL_H

namespace chess {

// Util class to compute once (when needed).
template <typename T>
class LazyGet {
 public:
  template <typename Func>
  T& Get(Func generator) {
    if (fetched_) {
      return data_;
    }

    data_ = generator();
    fetched_ = true;
    return data_;
  }

 private:
  bool fetched_ = false;
  T data_;
};

void PrintStackTrace();

}  // namespace chess

#endif
