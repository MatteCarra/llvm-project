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
#include <iostream>
#include <ostream>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static void addDashXForInput(const ArgList &Args, const InputInfo &Input, ArgStringList &CmdArgs) {
    //TODO is this needed?
    //CmdArgs.push_back(types::getTypeName(Input.getType()));
}

void Marco::addMarcoOptions(const ArgList &Args,
                                     ArgStringList &CmdArgs) const {
    Args.AddAllArgs(
      CmdArgs,
      {
        options::OPT_g_Flag,
        options::OPT_omc_path,
        options::OPT_omc_arg,
        options::OPT_omc_bypass,
        options::OPT_assertions,
        options::OPT_no_assertions,
        options::OPT_bit_width,
        options::OPT_function_inlining,
        options::OPT_no_function_inlining,
        options::OPT_output_arrays_promotion,
        options::OPT_no_output_arrays_promotion,
        options::OPT_read_only_variables_propagation,
        options::OPT_no_read_only_variables_propagation,
        options::OPT_variables_to_parameters_promotion,
        options::OPT_no_variables_to_parameters_promotion,
        options::OPT_cse,
        options::OPT_no_cse,
        options::OPT_omp,
        options::OPT_no_omp,
        options::OPT_O,
        options::OPT_target,
        options::OPT_mcpu_EQ,
        options::OPT_model,
        options::OPT_filter,
        options::OPT_solver,
        options::OPT_ida_reduced_system,
        options::OPT_no_ida_reduced_system,
        options::OPT_ida_reduced_derivatives,
        options::OPT_no_ida_reduced_derivatives,
        options::OPT_ida_jacobian_one_sweep,
        options::OPT_no_ida_jacobian_one_sweep,
        options::OPT_init_only
     }
    );
}

void Marco::ConstructJob(Compilation &C, const JobAction &JA,
                         const InputInfo &Output, const InputInfoList &Inputs,
                         const ArgList &Args, const char *LinkingOutput) const {
  const auto &TC = getToolChain();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  const std::string &TripleStr = Triple.getTriple();

  std::cout << "Job " << TripleStr << std::endl;

  const Driver &D = TC.getDriver();
  ArgStringList CmdArgs;

  // Invoke ourselves in -mc1 mode.
  CmdArgs.push_back("-mc1");

  // Add the "effective" target triple.
  //CmdArgs.push_back("-triple");
  //CmdArgs.push_back(Args.MakeArgString(TripleStr));

  //TODO how to handle mlir?
  if (isa<CompileJobAction>(JA) || isa<BackendJobAction>(JA)) {
    if(JA.getType() == types::TY_Modelica) {
      if(Args.hasArg(options::OPT_emit_modelica_flattened)) {
        CmdArgs.push_back("-emit-flattened");
      } else {
        assert(false && "Unexpected output type!");
      }
    } else if (JA.getType() == types::TY_AST) {
      if(Args.hasArg(options::OPT_emit_final_ast)) {
        CmdArgs.push_back("-emit-ast");
      } else {
        CmdArgs.push_back("-emit-final-ast");
      }
    } else if (JA.getType() == types::TY_LLVM_IR || JA.getType() == types::TY_LTO_IR) {
      CmdArgs.push_back("-emit-llvm");
    } else if (JA.getType() == types::TY_LLVM_BC || JA.getType() == types::TY_LTO_BC) {
      CmdArgs.push_back("-emit-llvm-bc");
    } else {
      assert(false && "Unexpected output type!");
    }
  } else if (isa<AssembleJobAction>(JA)) {
    CmdArgs.push_back("-emit-obj");
  } else {
    assert(false && "Unexpected action class for Marco tool.");
  }

  addMarcoOptions(Args, CmdArgs);

  // Color diagnostics are parsed by the driver directly from argv and later
  // re-parsed to construct this job; claim any possible color diagnostic here
  // to avoid warn_drv_unused_argument.
  Args.getLastArg(options::OPT_fcolor_diagnostics,
                  options::OPT_fno_color_diagnostics);
  
  if (D.getDiags().getDiagnosticOptions().ShowColors)
    CmdArgs.push_back("-fcolor-diagnostics");

  // Forward -mllvm options to the LLVM option parser. In practice, this means
  // forwarding to `-mc1` as that's where the LLVM parser is run.
  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    A->claim();
    A->render(Args, CmdArgs);
  }

  for (const Arg *A : Args.filtered(options::OPT_mmlir)) {
    A->claim();
    A->render(Args, CmdArgs);
  }

  // Optimization level for CodeGen.
  if (const Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O4)) {
      CmdArgs.push_back("-O3");
      D.Diag(diag::warn_O4_is_O3);
    } else {
      A->render(Args, CmdArgs);
    }
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  const InputInfo &Input = Inputs[0];
  assert(Input.isFilename() && "Invalid input.");

  for(auto& input : Inputs) {
    std::cout << "Input " << input.getFilename() << "." << std::endl;
  }

  CmdArgs.push_back(Input.getFilename());

  const char *Exec = Args.MakeArgString(D.GetProgramPath("marco", TC));
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileUTF8(),
                                         Exec, CmdArgs, Inputs, Output));
}

Marco::Marco(const ToolChain &TC) : Tool("marco", "marco frontend", TC) {}

Marco::~Marco() {}
