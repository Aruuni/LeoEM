#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x2a25f470, "tcp_slow_start" },
	{ 0x7f02188f, "__msecs_to_jiffies" },
	{ 0x557b4cf9, "tcp_cong_avoid_ai" },
	{ 0x122c3a7e, "_printk" },
	{ 0x54b1fac6, "__ubsan_handle_load_invalid_value" },
	{ 0x944067cc, "init_net" },
	{ 0xbd5f0a12, "__netlink_kernel_create" },
	{ 0x43bbf9b5, "tcp_register_congestion_control" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0x991bef01, "tcp_reno_undo_cwnd" },
	{ 0x819a5095, "param_ops_int" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x37befc70, "jiffies_to_msecs" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0xa648e561, "__ubsan_handle_shift_out_of_bounds" },
	{ 0xb8b79ff2, "tcp_unregister_congestion_control" },
	{ 0x3cd05b26, "netlink_kernel_release" },
	{ 0xd1a32cdf, "module_layout" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "BDA211DA275456B1933A1F8");
