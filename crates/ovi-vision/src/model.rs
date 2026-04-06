/// Target inference device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
    Auto,
}

impl Device {
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "CPU",
            Device::Gpu => "GPU",
            Device::Auto => "AUTO",
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_as_str() {
        assert_eq!(Device::Cpu.as_str(), "CPU");
        assert_eq!(Device::Gpu.as_str(), "GPU");
        assert_eq!(Device::Auto.as_str(), "AUTO");
    }

    #[test]
    fn test_device_default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }
}
