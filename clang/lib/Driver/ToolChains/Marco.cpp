//===-- Flang.cpp - Flang+LLVM ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "Marco.h"
#include "CommonArgs.h"

#include "clang/Driver/Options.h"

#include <cassert>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static void addDashXForInput(const ArgList &Args, const InputInfo &Input, ArgStringList &CmdArgs) {
    //TODO
}

void Marco::addMarcoDialectOptions(const ArgList &Args,
                                     ArgStringList &CmdArgs) const {
    //TODO
}

void Marco::addPreprocessingOptions(const ArgList &Args, ArgStringList &CmdArgs) const {
  //TODO
}

void Marco::addOtherOptions(const ArgList &Args, ArgStringList &CmdArgs) const {
  //TODO
}

void Marco::addPicOptions(const ArgList &Args, ArgStringList &CmdArgs) const {
  //TODO
}

static void addFloatingPointOptions(const Driver &D, const ArgList &Args, ArgStringList &CmdArgs) {
    //TODO
}

void Marco::ConstructJob(Compilation &C, const JobAction &JA,
                         const InputInfo &Output, const InputInfoList &Inputs,
                         const ArgList &Args, const char *LinkingOutput) const {
    //TODO
}

Marco::Marco(const ToolChain &TC) : Tool("marco", "marco frontend", TC) {}

Marco::~Marco() {}
