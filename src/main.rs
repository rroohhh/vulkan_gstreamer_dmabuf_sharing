use anyhow::{anyhow, Result};
use core::mem;
use gstreamer::{prelude::*, Buffer, BufferMap, Format, Fraction, Memory, ParseContext, Pipeline};
use gstreamer_app::AppSrc;
use gstreamer_allocators::DmaBufAllocator;
use gstreamer_video::{VideoFormat, VideoFrameRef, VideoInfo, VideoMeta, VIDEO_MAX_PLANES, VideoFrameFlags};
use std::{
    mem::MaybeUninit,
    any::Any,
    io::Write,
    sync::{Arc, MutexGuard},
    thread::{spawn, JoinHandle},
};
use vulkano_shaders;
use vulkano::{
    instance::{Instance, InstanceExtensions, PhysicalDevice},
    buffer::{DeviceLocalBuffer, BufferUsage, CpuAccessibleBuffer, DmaBufBuffer, TypedBufferAccess},
    command_buffer::AutoCommandBufferBuilder,
    descriptor::{
        descriptor_set::PersistentDescriptorSet,
        pipeline_layout::PipelineLayout,
        PipelineLayoutAbstract,
    },
    device::{Device, Queue, DeviceExtensions},
    pipeline::ComputePipeline,
    sync,
    sync::GpuFuture,
};

fn main() -> Result<()> {
    gstreamer::init()?;
    let mut context = ParseContext::new();
    let pipeline_string =
        "appsrc ! vaapipostproc ! vaapih264enc ! filesink location=test.h264";
    let pipeline = gstreamer::parse_launch_full(
        &pipeline_string,
        Some(&mut context),
        gstreamer::ParseFlags::empty(),
    )?
    .dynamic_cast::<Pipeline>()
    .unwrap();

    let appsrc = pipeline
        .get_children()
        .into_iter()
        .last()
        .unwrap()
        .dynamic_cast::<AppSrc>()
        .unwrap();

    let thread_handle = Some(spawn(move || {
        main_loop(pipeline).unwrap();
    }));

    let allocator = DmaBufAllocator::new();

    let width = 4096;
    let height = 3072;

    let video_info = VideoInfo::builder(VideoFormat::Rgbx, width as u32, height as u32)
        .fps(Fraction::new(30, 1))
        .build()
        .expect("Failed to create video info");
    appsrc.set_caps(Some(&video_info.to_caps().unwrap()));
    appsrc.set_property_format(Format::Time);

    // let width = 1920;
    // let height = 1080;
    let mut t = 0;

    let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_compute())
        .unwrap();

    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &DeviceExtensions {
            khr_8bit_storage: true,
            ext_external_memory_fd: true,
            ext_external_memory_dma_buf: true,
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        },
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();
    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
            #version 450
            #extension GL_EXT_shader_explicit_arithmetic_types: enable
            #extension GL_EXT_shader_explicit_arithmetic_types_int8: require

            layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } data;

            layout(push_constant) uniform PushConstantData {
                uint width;
                uint height;
                uint frame;
            } params;

            void main() {
                uint x = gl_GlobalInvocationID.x;
                uint y = gl_GlobalInvocationID.y;
                data.data[(y * params.width) + x] = ((x & 0xff) << 16) | ((y & 0xff) << 8) | ((params.frame & 0xff) << 0);
                // data.data[4 * (y * params.width + x) + 0] = uint8_t(x);
                // data.data[4 * (y * params.width + x) + 1] = uint8_t(y);
                // data.data[4 * (y * params.width + x) + 2] = uint8_t(params.frame);
                // data.data[4 * (y * params.width + x) + 3] = uint8_t(0);
                // data.data[4 * (y * params.width + x) + 0] = uint8_t(255);
                // data.data[4 * (y * params.width + x) + 1] = uint8_t(255);
                // data.data[4 * (y * params.width + x) + 2] = uint8_t(255);
                // data.data[4 * (y * params.width + x) + 3] = uint8_t(255);
            }
        "
        }
    }

    let pipeline = Arc::new({
        let shader = cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &(), None).unwrap()
    });

    let mut data_buffers: Vec<(Arc<DmaBufBuffer>, i32)> = vec![];

    for i in 0..50 {
        let data_buffer =
            DmaBufBuffer::new(device.clone(), width * height * 4, BufferUsage::all()).unwrap();
        // let fd = data_buffer.leak_fd().unwrap();
        // unsafe { libc::close(fd) };
        data_buffers.push((data_buffer.clone(), -1));
    }


    // let data_buffer = DeviceLocalBuffer::<[u8]>::array(device.clone(), width * height * 4, BufferUsage::all(), std::iter::once(queue_family)).unwrap();
        // let data_buffer =
        //     DmaBufBuffer::new(device.clone(), width * height * 4, BufferUsage::all()).unwrap();
    loop {
        t += 1;

        // let data_buffer =
        //     DmaBufBuffer::new(device.clone(), width * height * 4, BufferUsage::all()).unwrap();

        let mut s = MaybeUninit::uninit();
        let mut data_buffer = None;

        while let None = data_buffer {
            let mut the_fd = None;
            for (i, buffer) in data_buffers.iter_mut().enumerate() {
                unsafe {
                    let ret = libc::fstat(buffer.1, s.as_mut_ptr());
                    // println!("{} fd {} is {}",i,  buffer.1, ret);
                    if ret == -1 {
                        the_fd = Some((i, buffer.1));
                        let fd =  buffer.0.leak_fd().unwrap();
                        println!("recycling {:?} {:?}", i, fd);
                        *buffer = (buffer.0.clone(), fd);
                        data_buffer = Some(buffer.clone());
                        break;
                    }
                }
            }

            if let Some((j, the_fd)) = the_fd {
                for (i, buffer) in data_buffers.iter_mut().enumerate() {
                    if (buffer.1 == the_fd) && (i != j) {
                        println!("invalidating {}", buffer.1);
                        buffer.1 = -1;
                    }
                }
            }

            // println!("sleeping");
            // std::thread::sleep(std::time::Duration::from_nanos(10_000));
        }

        let (data_buffer, fd) = data_buffer.unwrap();

        let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(data_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let push_constants = cs::ty::PushConstantData {
            width: width as u32,
            height: height as u32,
            frame: t,
        };

        let mut builder =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap();

        builder
            .dispatch([(width / 32) as u32, (height / 32) as u32, 1], pipeline.clone(), set, push_constants)
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();
        println!("pushing frame {}", t);

        // let buffer = Buffer::from_slice((&*data_buffer.read().unwrap()).to_vec());

        // let fd = data_buffer.leak_fd().unwrap();
        let mut buffer = Buffer::new();
        let mem = allocator.alloc(fd, width * height * 4).unwrap();
        {
            let buffer = buffer.make_mut();
            buffer.append_memory(mem);
            // VideoMeta::add_full(buffer, VideoFrameFlags::empty(), VideoFormat::Rgbx, 1920, 1080, &[0; VIDEO_MAX_PLANES], &[0; VIDEO_MAX_PLANES]).unwrap();
        }
        appsrc.push_buffer(buffer)?;

        if t >= 500 {
            appsrc.end_of_stream().unwrap();
            drop(appsrc);
            break;
        }
    }

    // thread_handle.unwrap().join();

    Ok(())
}

fn main_loop(pipeline: gstreamer::Pipeline) -> Result<()> {
    pipeline.set_state(gstreamer::State::Playing)?;

    let bus = pipeline
        .get_bus()
        .expect("Pipeline without bus. Shouldn't happen!");

    for msg in bus.iter_timed(gstreamer::CLOCK_TIME_NONE) {
        use gstreamer::MessageView;

        println!("{:?}", msg);

        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                pipeline.set_state(gstreamer::State::Null)?;
                return Err(anyhow!(
                    "{:?}{:?}{:?}{:?}",
                    msg.get_src()
                        .map(|s| String::from(s.get_path_string()))
                        .unwrap_or_else(|| String::from("None")),
                    err.get_error().to_string(),
                    err.get_debug(),
                    err.get_error()
                ));
            }
            _ => (),
        }
    }

    pipeline.set_state(gstreamer::State::Null)?;

    Ok(())
}
