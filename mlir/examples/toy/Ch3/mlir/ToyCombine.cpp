//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "toy/Dialect.h"
#include <numeric>
using namespace mlir;
using namespace toy;

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult matchAndRewrite(TransposeOp op, mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

struct RedundantReshapeOptPattern : public ::mlir::OpRewritePattern<ReshapeOp> {
  RedundantReshapeOptPattern(::mlir::MLIRContext *context): ::mlir::OpRewritePattern<ReshapeOp>(context) {

  }

  ::mlir::LogicalResult matchAndRewrite(ReshapeOp op0, ::mlir::PatternRewriter &rewriter) const override {
    if (!(op0.getResult().getType() == op0.input().getType())) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res, arg' failed to satisfy constraint: ";
      });
    }

    // Rewrite
    //auto odsLoc = rewriter.getFusedLoc({ op0->getLoc() });
    rewriter.replaceOp(op0, {op0.input()});
    return ::mlir::success();
  };
};

struct ReshapeReshapeOptPattern: public ::mlir::OpRewritePattern<ReshapeOp> {
  ReshapeReshapeOptPattern(::mlir::MLIRContext *context): ::mlir::OpRewritePattern<ReshapeOp>(context) {

  }

  ::mlir::LogicalResult matchAndRewrite(ReshapeOp op0, ::mlir::PatternRewriter &rewriter) const override {
    ReshapeOp input = op0.input().getDefiningOp<ReshapeOp>();
    if(!input)
      return failure();

    auto odsLoc = rewriter.getFusedLoc({op0.getLoc(), input.getLoc()});

    auto newOp = rewriter.create<::mlir::toy::ReshapeOp>(odsLoc, op0->getResultTypes(), input.input());

    rewriter.replaceOp(op0, { newOp });
    return mlir::success();
  }
};

struct FoldConstantReshapeOptPattern: public ::mlir::OpRewritePattern<ReshapeOp> {
  FoldConstantReshapeOptPattern(::mlir::MLIRContext *context): ::mlir::OpRewritePattern<ReshapeOp>(context) {
  }

  ::mlir::LogicalResult matchAndRewrite(ReshapeOp op0, ::mlir::PatternRewriter &rewriter) const override {
    ConstantOp input = op0.input().getDefiningOp<ConstantOp>();
    if(!input)
      return failure();

    if(input.getType() == op0.getResult().getType())
      return failure();

    auto location = rewriter.getFusedLoc({ op0->getLoc(), input->getLoc() });
    auto newOp = rewriter.create<ConstantOp>(location, input.value().reshape(op0.getResult().getType().cast<ShapedType>()));
    rewriter.replaceOp(op0, { newOp });

    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReshapeReshapeOptPattern, RedundantReshapeOptPattern, FoldConstantReshapeOptPattern>(context);
}